import copy
import math
import time
import torch
from collections import defaultdict
import torch.nn.functional as F
from typing import TYPE_CHECKING, Dict, Optional

from mpail2.envs.isaac.omnireset_wrapper import compact_omnireset_policy_obs
from mpail2.encoder import Coder
from mpail2.dynamics import Dynamics
from mpail2.reward import Reward
from mpail2.value import EnsembleValue
from mpail2.sampling import PolicyNetwork
from mpail2.planner import Planner
from .storage import RolloutStorage as Storage, DictDemoDataset
from .utils import resolve_optim, compute_disc_train_stats, resolve_obj
from .utils.obs_normalizer import ObsNormalizerFactory

if TYPE_CHECKING:
    from .configs.cfgs import  MPAIL2LearnerCfg, ValueLearnerCfg


class MPAIL2Learner:
    '''Composes MPPIPlanner to include buffer population for MPAIL training as well as exploratory
    action selection.'''

    planner: Planner
    _encoder: Coder
    _dynamics: Dynamics
    _reward: Reward
    _value: EnsembleValue
    _policy: PolicyNetwork

    def __init__(self,
        demonstrations: torch.Tensor | Storage,
        num_envs: int,
        learner_config: 'MPAIL2LearnerCfg',
        device: str = "cuda",
        dtype=torch.float32
    ):

        self.device = device
        self.dtype = dtype

        self.cfg = learner_config
        self.value_learner_cfg = self.cfg.value_learner_cfg
        self.reward_learner_cfg = self.cfg.reward_learner_cfg

        # Log important settings
        print(f"[MPAIL2Learner] use_terminations: {self.cfg.use_terminations}")

        self.transition = Storage.Transition()

        # Learning params
        self._demonstrations = demonstrations
        self.replay_ratio = self.cfg.replay_ratio

        #
        # PLANNER SETUP
        #

        self.planner:Planner = Planner(self.cfg.planner_cfg, num_envs, device=device, dtype=dtype)

        self._encoder:Coder = self.planner.encoder
        self._dynamics:Dynamics = self.planner.dynamics
        self._reward:Reward = self.planner.reward
        self._reward_center = torch.zeros((), device=self.device, dtype=self.dtype)
        self.planner.cfg.reward_center_for_planning = (
            self.planner.cfg.reward_center_for_planning or self.cfg.reward_center_for_planning
        )
        if self.cfg.reward_clip_for_planning is not None:
            self.planner.cfg.reward_clip_for_planning = self.cfg.reward_clip_for_planning
        self._value:EnsembleValue = self.planner.value
        self._policy:PolicyNetwork = self.planner.sampling.policy
        self._raw_base_policy = None
        self._raw_base_policy_mean = None
        self._raw_base_policy_std = None
        self._raw_base_policy_obs_key = getattr(self.cfg, "raw_base_policy_obs_key", "raw_policy")
        if getattr(self.cfg, "raw_base_policy_checkpoint", None):
            self._load_raw_base_policy(self.cfg.raw_base_policy_checkpoint)

        self._decoder:Optional[Coder] = self.planner.decoder

        #
        ## REWARD SETUP
        #

        self._reward_opt = resolve_optim(self.reward_learner_cfg.opt)(
            self._reward.parameters(),
            **self.reward_learner_cfg.opt_params
        )

        #
        ## VALUE SETUP
        #

        self._value_opt = resolve_optim(self.value_learner_cfg.opt)(
            self._value.parameters(),
            **self.value_learner_cfg.opt_params
        )

        # Setup Polyak averaging for value networks
        self._polyak_tau = self.value_learner_cfg.polyak_tau

        # Create target networks as deep copies of the main networks
        self._value_target = self._create_target_value_network()

        #
        # DYNAMICS SETUP
        #

        self.dynamics_learner_cfg = self.cfg.dynamics_learner_cfg

        _enc_lr = self.dynamics_learner_cfg.opt_params['lr'] * self.dynamics_learner_cfg.enc_lr_scale
        dyn_params = [
            {"params": self._encoder.parameters(), "lr": _enc_lr},
            {"params": self._dynamics.parameters()},
        ]
        if self._decoder is not None:
            dyn_params.append({"params": self._decoder.parameters()})
        self._dyn_opt = resolve_optim(self.dynamics_learner_cfg.opt)(
            dyn_params,
            **self.dynamics_learner_cfg.opt_params
        )

        #
        # POLICY SAMPLING SETUP
        #

        self.policy_learner_cfg = self.cfg.policy_learner_cfg
        self._policy_opt = resolve_optim(self.policy_learner_cfg.opt)(
            self._policy.parameters(),
            **self.policy_learner_cfg.opt_params
        )
        self._residual_adapter = None
        self._residual_adapter_opt = None
        self._candidate_selector = None
        self._candidate_selector_opt = None
        if getattr(self.cfg, "residual_adapter_enabled", False) or getattr(self.cfg, "residual_adapter_apply", False):
            hidden_dim = int(getattr(self.cfg, "residual_adapter_hidden_dim", 256))
            input_dim = (
                int(getattr(self.cfg, "raw_base_policy_obs_dim", 215))
                if getattr(self.cfg, "residual_adapter_use_raw_obs", False)
                else int(self.cfg.planner_cfg.latent_dim)
            )
            action_dim = int(self.cfg.planner_cfg.action_dim)
            self._residual_adapter = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, action_dim),
            ).to(device=self.device, dtype=self.dtype)
            self._residual_adapter_opt = torch.optim.Adam(
                self._residual_adapter.parameters(),
                lr=float(getattr(self.cfg, "residual_adapter_lr", 1e-4)),
            )
        if getattr(self.cfg, "mppi_candidate_selector_enabled", False) or getattr(
            self.cfg, "mppi_candidate_selector_apply", False
        ):
            hidden_dim = int(getattr(self.cfg, "mppi_candidate_selector_hidden_dim", 256))
            latent_dim = int(self.cfg.planner_cfg.latent_dim)
            action_dim = int(self.cfg.planner_cfg.action_dim)
            self._candidate_selector = torch.nn.Sequential(
                torch.nn.Linear(latent_dim + action_dim * 3, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, 1),
            ).to(device=self.device, dtype=self.dtype)
            self._candidate_selector_opt = torch.optim.Adam(
                self._candidate_selector.parameters(),
                lr=float(getattr(self.cfg, "mppi_candidate_selector_lr", 1e-4)),
            )

        # Initialize log_alpha as a learnable parameter
        self._log_alpha = torch.tensor(math.log(1.0), requires_grad=True, device=self.device)
        self._alpha_opt = torch.optim.Adam([self._log_alpha], lr=self.policy_learner_cfg.alpha_lr)
        self._target_entropy = self.policy_learner_cfg.target_entropy

        #
        # OBSERVATION NORMALIZER SETUP
        #

        self._obs_normalizer = None
        if self.cfg.obs_normalizer_cfg is not None:
            self._obs_normalizer = ObsNormalizerFactory.create_normalizer(
                normalization_type=self.cfg.obs_normalizer_cfg.normalization_type,
                device=self.device,
                demonstrations=self._demonstrations,  # Pass demos for fixed normalizer
                eps=self.cfg.obs_normalizer_cfg.eps,
                std_floor=self.cfg.obs_normalizer_cfg.std_floor,
                skip_low_std_dims=self.cfg.obs_normalizer_cfg.skip_low_std_dims,
            )

            self.planner.set_obs_normalizer(self._obs_normalizer)

        self._bc_obs = None
        self._bc_next_obs = None
        self._bc_actions = None
        self._iql_obs = None
        self._iql_next_obs = None
        self._iql_actions = None
        self._iql_not_done = None
        self._online_mppi_rank_buffer = None
        self._online_mppi_outcome_rank_buffer = None
        self._online_mppi_candidate_rank_buffer = None
        self._mppi_elite_adversarial_buffer = None
        self._act_calls = 0
        if self.cfg.demo_action_bc_path and (
            self.cfg.demo_action_bc_coeff > 0 or self.cfg.demo_action_dyn_coeff > 0
            or getattr(self.cfg, "demo_action_multistep_dyn_coeff", 0.0) > 0
            or getattr(self.cfg, "demo_action_bc_direct_steps", 0) > 0
            or getattr(self.cfg, "mppi_action_library_size", 0) > 0
            or getattr(self.planner.reward.cfg, "use_action", False)
        ):
            self._load_demo_action_bc(self.cfg.demo_action_bc_path)
            if self.cfg.demo_action_bc_pretrain_steps > 0:
                self._pretrain_policy_on_demo_actions(
                    steps=self.cfg.demo_action_bc_pretrain_steps,
                    batch_size=self.cfg.demo_action_bc_pretrain_batch_size,
                )
            if self.cfg.demo_action_dyn_pretrain_steps > 0:
                self._pretrain_dynamics_on_demo_actions(
                    steps=self.cfg.demo_action_dyn_pretrain_steps,
                    batch_size=self.cfg.demo_action_dyn_pretrain_batch_size,
                )
        if getattr(self.cfg, "iql_pretrain_path", None) and int(getattr(self.cfg, "iql_pretrain_steps", 0)) > 0:
            self._load_iql_demo_transitions(self.cfg.iql_pretrain_path)
            self._pretrain_iql_on_demo_transitions(
                steps=int(self.cfg.iql_pretrain_steps),
                batch_size=int(getattr(self.cfg, "iql_pretrain_batch_size", 512)),
            )
        if getattr(self.cfg, "iqlearn_pretrain_path", None) and int(getattr(self.cfg, "iqlearn_pretrain_steps", 0)) > 0:
            self._load_iql_demo_transitions(self.cfg.iqlearn_pretrain_path)
            self._pretrain_iqlearn_on_demo_transitions(
                steps=int(self.cfg.iqlearn_pretrain_steps),
                batch_size=int(getattr(self.cfg, "iqlearn_pretrain_batch_size", 512)),
            )
            distill_steps = int(getattr(self.cfg, "iqlearn_reward_distill_steps", 0))
            if distill_steps > 0:
                self._distill_iqlearn_reward(
                    steps=distill_steps,
                    batch_size=int(getattr(self.cfg, "iqlearn_reward_distill_batch_size", 512)),
                )
        if getattr(self.cfg, "demo_success_replay_path", None) and self._iql_obs is None:
            self._load_iql_demo_transitions(self.cfg.demo_success_replay_path)

        self.planner.reset()

    def _load_raw_base_policy(self, checkpoint_path: str):
        sampling_cfg = self.cfg.planner_cfg.sampling_cfg
        action_dim = int(self.cfg.planner_cfg.action_dim)
        horizon = int(sampling_cfg.num_timesteps)
        obs_dim = int(getattr(self.cfg, "raw_base_policy_obs_dim", 215))
        model = resolve_obj(sampling_cfg.model_factory)(
            input_dim=obs_dim,
            output_dim=horizon * action_dim * 2,
            **sampling_cfg.model_kwargs,
        )
        policy = PolicyNetwork(
            model=model,
            min_std=sampling_cfg.min_std,
            max_std=sampling_cfg.max_std,
            horizon=horizon,
            action_dim=action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        saved = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = saved["model_state_dict"] if isinstance(saved, dict) and "model_state_dict" in saved else saved
        prefix = "sampling.policy."
        policy_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        policy.load_state_dict(policy_state)
        policy.eval()
        for param in policy.parameters():
            param.requires_grad_(False)
        self._raw_base_policy = policy.to(device=self.device, dtype=self.dtype)
        norm_demo_path = getattr(self.cfg, "raw_base_policy_normalizer_demo_path", None)
        if norm_demo_path:
            demo = torch.load(norm_demo_path, map_location="cpu", weights_only=False)
            if isinstance(demo, dict) and "demonstrations" in demo:
                raw_demo = demo["demonstrations"]["policy"].reshape(-1, obs_dim).float()
            elif isinstance(demo, dict) and "data" in demo:
                obs = demo["data"]["obs"]["policy"].float()
                next_obs = demo["data"]["next_obs"]["policy"].float()
                raw_demo = torch.cat([obs, next_obs], dim=0)
            else:
                raise ValueError(f"Unsupported raw base normalizer demo format: {norm_demo_path}")
            mean = raw_demo.mean(dim=0).to(device=self.device, dtype=self.dtype)
            std = raw_demo.std(dim=0).to(device=self.device, dtype=self.dtype)
            std_floor = float(getattr(self.cfg, "raw_base_policy_normalizer_std_floor", 0.03))
            if std_floor > 0:
                std = torch.clamp(std, min=std_floor)
            self._raw_base_policy_mean = mean
            self._raw_base_policy_std = std
            print(
                f"[MPAIL2Learner] Raw base policy normalizer: {norm_demo_path} "
                f"std_min={std.min().item():.4f}",
                flush=True,
            )
        print(f"[MPAIL2Learner] Loaded raw base policy: {checkpoint_path}", flush=True)

    def init_storage(
        self,
        num_envs,
        num_steps_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        # create rollout storage
        self.storage = Storage(
            num_envs,
            num_steps_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            replay_size=self.cfg.replay_size,
            replay_batch_size=self.cfg.replay_batch_size,
            device=self.device,
            obs_normalizer=self._obs_normalizer,
            protect_demo_replay=bool(getattr(self.cfg, "protect_demo_success_replay", True)),
        )
        self._seed_demo_success_replay()

    def _seed_demo_success_replay(self):
        if not getattr(self.cfg, "demo_success_replay_path", None):
            return
        if self._iql_obs is None or self._iql_actions is None or self._iql_next_obs is None:
            return
        fraction = float(getattr(self.cfg, "demo_success_replay_fraction", 0.0))
        if fraction <= 0.0:
            return
        max_trajs = max(1, int(self.storage.replay_dim_size * min(fraction, 1.0)))
        added = self.storage.add_demo_transitions_to_replay(
            obs={"policy": self._iql_obs},
            next_obs={"policy": self._iql_next_obs},
            actions=self._iql_actions,
            not_done=self._iql_not_done,
            max_trajectories=max_trajs,
        )
        print(
            f"[MPAIL2Learner] Seeded protected demo-success replay: "
            f"traj_slots={added}/{self.storage.replay_dim_size} "
            f"transitions={self._iql_actions.shape[0]} protect={bool(getattr(self.cfg, 'protect_demo_success_replay', True))}",
            flush=True,
        )

    def eval(self):
        self.planner.eval()

    def train(self):
        self.planner.train()

    def _load_demo_action_bc(self, path: str):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        obs = data["obs"]["policy"].float()
        next_obs = data["next_obs"]["policy"].float()
        actions = data["actions"].float()

        target_obs_dim = next(iter(self._demonstrations.values())).shape[-1]
        if obs.shape[-1] != target_obs_dim:
            if obs.shape[-1] == 215 and target_obs_dim in (13, 19, 22, 60, 119):
                if target_obs_dim in (60, 119):
                    from mpail2.envs.isaac.omnireset_wrapper import (
                        planner_v2_omnireset_policy_obs,
                        planner_v3_omnireset_policy_obs,
                    )
                    transform = planner_v3_omnireset_policy_obs if target_obs_dim == 119 else planner_v2_omnireset_policy_obs
                    obs = transform(obs)
                    next_obs = transform(next_obs)
                    transform_kwargs = None
                else:
                    transform_kwargs = {
                        "include_ee_to_hole": target_obs_dim in (19, 22),
                        "include_ee_vel": target_obs_dim == 22,
                    }
                if transform_kwargs is None:
                    pass
                else:
                    obs = compact_omnireset_policy_obs(
                        obs,
                        **transform_kwargs,
                    )
                    next_obs = compact_omnireset_policy_obs(
                        next_obs,
                        **transform_kwargs,
                    )
            else:
                raise ValueError(
                    f"Cannot adapt BC obs dim {obs.shape[-1]} to learner obs dim {target_obs_dim}"
                )

        raw_limit = self.cfg.demo_action_raw_limit
        if raw_limit is None:
            raise ValueError("demo_action_raw_limit must be set when demo_action_bc_path is used")
        raw_limit_t = torch.as_tensor(raw_limit, dtype=torch.float32).view(1, -1).clamp_min(1e-6)
        actions = (actions / raw_limit_t).clamp(-1.0, 1.0)

        self._bc_obs = obs.to(device=self.device, dtype=self.dtype)
        self._bc_next_obs = next_obs.to(device=self.device, dtype=self.dtype)
        self._bc_actions = actions.to(device=self.device, dtype=self.dtype)
        print(
            f"[MPAIL2Learner] Loaded BC action prior: {self._bc_obs.shape[0]} samples "
            f"bc_coeff={self.cfg.demo_action_bc_coeff} dyn_coeff={self.cfg.demo_action_dyn_coeff}",
            flush=True,
        )

    def _adapt_demo_policy_obs(self, obs: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_obs_dim = next(iter(self._demonstrations.values())).shape[-1]
        if obs.shape[-1] == target_obs_dim:
            return obs, next_obs
        if obs.shape[-1] == 215 and target_obs_dim in (13, 19, 22, 60, 119):
            if target_obs_dim in (60, 119):
                from mpail2.envs.isaac.omnireset_wrapper import (
                    planner_v2_omnireset_policy_obs,
                    planner_v3_omnireset_policy_obs,
                )
                transform = planner_v3_omnireset_policy_obs if target_obs_dim == 119 else planner_v2_omnireset_policy_obs
                return transform(obs), transform(next_obs)
            transform_kwargs = {
                "include_ee_to_hole": target_obs_dim in (19, 22),
                "include_ee_vel": target_obs_dim == 22,
            }
            return (
                compact_omnireset_policy_obs(obs, **transform_kwargs),
                compact_omnireset_policy_obs(next_obs, **transform_kwargs),
            )
        raise ValueError(f"Cannot adapt demo obs dim {obs.shape[-1]} to learner obs dim {target_obs_dim}")

    def _normalize_demo_actions(self, actions: torch.Tensor) -> torch.Tensor:
        raw_limit = self.cfg.demo_action_raw_limit
        if raw_limit is None:
            raise ValueError("demo_action_raw_limit must be set when loading raw demo actions")
        raw_limit_t = torch.as_tensor(raw_limit, dtype=torch.float32).view(1, -1).clamp_min(1e-6)
        return (actions / raw_limit_t).clamp(-1.0, 1.0)

    def _load_iql_demo_transitions(self, path: str):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        obs = data["obs"]["policy"].float()
        next_obs = data["next_obs"]["policy"].float()
        obs, next_obs = self._adapt_demo_policy_obs(obs, next_obs)
        actions = self._normalize_demo_actions(data["actions"].float())

        not_done = torch.ones(actions.shape[0], dtype=torch.float32)
        episode_id = data.get("episode_id", None) if isinstance(data, dict) else None
        if episode_id is not None and episode_id.numel() == actions.shape[0]:
            not_done[:-1] = (episode_id[:-1] == episode_id[1:]).float()
            not_done[-1] = 0.0

        self._iql_obs = obs.to(device=self.device, dtype=self.dtype)
        self._iql_next_obs = next_obs.to(device=self.device, dtype=self.dtype)
        self._iql_actions = actions.to(device=self.device, dtype=self.dtype)
        self._iql_not_done = not_done.to(device=self.device, dtype=self.dtype)
        configured_steps = max(
            int(getattr(self.cfg, "iql_pretrain_steps", 0)),
            int(getattr(self.cfg, "iqlearn_pretrain_steps", 0)),
        )
        print(
            f"[MPAIL2Learner] Loaded IQL demo transitions: {self._iql_obs.shape[0]} samples "
            f"steps={configured_steps}",
            flush=True,
        )

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return weight * diff.pow(2)

    def _policy_mean_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        out = self._policy.model(z)
        mu, _log_std = torch.chunk(out, 2, dim=-1)
        return torch.tanh(mu.view(z.shape[0], self._policy.horizon, self._policy.action_dim)[:, 0])

    def _sample_policy_first_action_and_logp(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        plans = self._policy.plan(z)
        actions = plans[:, 0, :]
        logp = self._policy.log_prob.squeeze(-1)
        return actions, logp

    def _iqlearn_alpha(self) -> torch.Tensor:
        return self._log_alpha.exp().squeeze()

    def _iqlearn_v_pi(self, z: torch.Tensor, *, target: bool = False, deterministic: bool = False) -> torch.Tensor:
        alpha = self._iqlearn_alpha().detach()
        with torch.no_grad():
            if deterministic:
                actions = self._policy_mean_from_latent(z)
                logp = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
            else:
                actions, logp = self._sample_policy_first_action_and_logp(z)
            actions = actions.detach()
            logp = logp.detach()
        value_net = self._value_target if target else self._value
        q = value_net(z, actions, return_type="min")
        return q - alpha * logp

    def _iqlearn_reward_from_latent(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        next_z: torch.Tensor,
        not_done: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gamma = float(self.value_learner_cfg.gamma)
        if not_done is None:
            not_done = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        deterministic = bool(getattr(self.cfg, "iqlearn_pretrain_deterministic_policy", False))
        next_v = self._iqlearn_v_pi(next_z, target=True, deterministic=deterministic).detach()
        q = self._value(z, actions, return_type="min")
        reward = q - gamma * not_done * next_v
        clip = getattr(self.cfg, "iqlearn_reward_clip", None)
        if clip is not None and float(clip) > 0.0:
            reward = reward.clamp(-float(clip), float(clip))
        return reward

    def _pretrain_iqlearn_on_demo_transitions(self, *, steps: int, batch_size: int):
        if self._iql_obs is None or self._iql_actions is None:
            return
        batch_size = min(int(batch_size), self._iql_obs.shape[0])
        was_training = self.planner.training
        self.planner.train()

        init_temp = getattr(self.cfg, "iqlearn_init_temp", None)
        if init_temp is not None and float(init_temp) > 0.0:
            with torch.no_grad():
                self._log_alpha.copy_(
                    torch.tensor(math.log(float(init_temp)), device=self.device, dtype=self._log_alpha.dtype)
                )
        alpha = max(float(getattr(self.cfg, "iqlearn_alpha", 1.0)), 1e-6)
        gamma = float(self.value_learner_cfg.gamma)
        dyn_coeff = float(getattr(self.cfg, "iqlearn_dynamics_coeff", 1.0))
        policy_ratio = max(float(getattr(self.cfg, "iqlearn_policy_batch_ratio", 1.0)), 0.0)
        policy_batch_size = max(int(round(batch_size * policy_ratio)), 1) if policy_ratio > 0.0 else 0
        regularize_all = bool(getattr(self.cfg, "iqlearn_regularize_all", True))
        learn_temp = bool(getattr(self.cfg, "iqlearn_learn_temp", False))
        deterministic_policy = bool(getattr(self.cfg, "iqlearn_pretrain_deterministic_policy", False))
        bc_coeff = float(getattr(self.cfg, "iqlearn_pretrain_bc_coeff", 0.0))
        log_every = max(int(getattr(self.cfg, "iqlearn_log_every", 1000)), 0)
        q_opt = torch.optim.Adam(
            [
                {"params": self._encoder.parameters(), "lr": self.dynamics_learner_cfg.opt_params["lr"] * self.dynamics_learner_cfg.enc_lr_scale},
                {"params": self._value.parameters(), "lr": self.value_learner_cfg.opt_params["lr"]},
            ]
        )
        last_stats = {}

        for step_idx in range(int(steps)):
            expert_idx = torch.randint(self._iql_obs.shape[0], (batch_size,), device=self.device)
            raw_expert_obs = {"policy": self._iql_obs[expert_idx]}
            raw_expert_next_obs = {"policy": self._iql_next_obs[expert_idx]}
            expert_obs = self._obs_normalizer(raw_expert_obs) if self._obs_normalizer is not None else raw_expert_obs
            expert_next_obs = (
                self._obs_normalizer(raw_expert_next_obs)
                if self._obs_normalizer is not None
                else raw_expert_next_obs
            )
            expert_actions = self._iql_actions[expert_idx]
            expert_not_done = self._iql_not_done[expert_idx]

            z_dyn = self._encoder(expert_obs)
            with torch.no_grad():
                next_z_dyn = self._encoder(expert_next_obs).detach()
            if dyn_coeff > 0.0:
                pred_next_z = self._dynamics(z0=z_dyn, controls=expert_actions.unsqueeze(1))[:, 1, :]
                dyn_loss = F.mse_loss(pred_next_z, next_z_dyn)
                self._dyn_opt.zero_grad(set_to_none=True)
                (dyn_coeff * dyn_loss).backward()
                torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), self.policy_learner_cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self._dynamics.parameters(), self.value_learner_cfg.max_grad_norm)
                self._dyn_opt.step()
            else:
                dyn_loss = torch.zeros((), device=self.device, dtype=self.dtype)

            expert_z = self._encoder(expert_obs)
            with torch.no_grad():
                expert_next_z = self._encoder(expert_next_obs).detach()

            if policy_batch_size > 0:
                policy_idx = torch.randint(self._iql_obs.shape[0], (policy_batch_size,), device=self.device)
                raw_policy_obs = {"policy": self._iql_obs[policy_idx]}
                policy_obs = self._obs_normalizer(raw_policy_obs) if self._obs_normalizer is not None else raw_policy_obs
                policy_z = self._encoder(policy_obs)
                with torch.no_grad():
                    if deterministic_policy:
                        policy_actions = self._policy_mean_from_latent(policy_z.detach())
                    else:
                        policy_actions, _policy_action_logp = self._sample_policy_first_action_and_logp(policy_z.detach())
                    policy_actions = policy_actions.detach()
                    policy_next_z = self._dynamics(z0=policy_z.detach(), controls=policy_actions.unsqueeze(1))[:, 1, :].detach()
                    policy_not_done = torch.ones(policy_batch_size, device=self.device, dtype=self.dtype)
                z = torch.cat([policy_z, expert_z], dim=0)
                next_z = torch.cat([policy_next_z, expert_next_z], dim=0)
                actions = torch.cat([policy_actions, expert_actions], dim=0)
                not_done = torch.cat([policy_not_done, expert_not_done], dim=0)
                is_expert = torch.cat(
                    [
                        torch.zeros(policy_batch_size, device=self.device, dtype=torch.bool),
                        torch.ones(batch_size, device=self.device, dtype=torch.bool),
                    ],
                    dim=0,
                )
            else:
                z = expert_z
                next_z = expert_next_z
                actions = expert_actions
                not_done = expert_not_done
                is_expert = torch.ones(batch_size, device=self.device, dtype=torch.bool)

            with torch.no_grad():
                next_v = self._iqlearn_v_pi(next_z, target=True, deterministic=deterministic_policy).detach()
                if deterministic_policy:
                    policy_actions_for_v = self._policy_mean_from_latent(z.detach())
                    policy_logp_for_v = torch.zeros(z.shape[0], device=self.device, dtype=self.dtype)
                else:
                    policy_actions_for_v, policy_logp_for_v = self._sample_policy_first_action_and_logp(z.detach())
                policy_actions_for_v = policy_actions_for_v.detach()
                policy_logp_for_v = policy_logp_for_v.detach()

            q_all = self._value(z, actions, return_type="all")
            current_alpha = self._iqlearn_alpha().detach()
            current_v = self._value(z, policy_actions_for_v, return_type="min") - current_alpha * policy_logp_for_v
            y = gamma * not_done * next_v
            reward_all = q_all - y.unsqueeze(0)
            expert_reward_all = reward_all[:, is_expert]
            softq_loss = -expert_reward_all.mean()
            value_loss = (current_v - y).mean()
            chi2_source = reward_all if regularize_all else expert_reward_all
            chi2_loss = chi2_source.pow(2).mean() / (4.0 * alpha)
            q_loss = softq_loss + value_loss + chi2_loss

            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), self.policy_learner_cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._value.parameters(), self.value_learner_cfg.max_grad_norm)
            q_opt.step()
            self._soft_update_target_value()

            z_pi = z.detach()
            if deterministic_policy:
                policy_actions = self._policy_mean_from_latent(z_pi)
                policy_logp = torch.zeros(z_pi.shape[0], device=self.device, dtype=self.dtype)
            else:
                policy_actions, policy_logp = self._sample_policy_first_action_and_logp(z_pi)
            policy_q = self._value(z_pi, policy_actions, return_type="min")
            actor_loss = (self._iqlearn_alpha().detach() * policy_logp - policy_q).mean()
            bc_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            if bc_coeff > 0.0:
                expert_policy_mean = self._policy_mean_from_latent(expert_z.detach())
                bc_loss = F.mse_loss(expert_policy_mean, expert_actions)
                actor_loss = actor_loss + bc_coeff * bc_loss

            self._policy_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self.policy_learner_cfg.max_grad_norm)
            self._policy_opt.step()

            alpha_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            if learn_temp:
                self._alpha_opt.zero_grad(set_to_none=True)
                alpha_loss = (
                    self._iqlearn_alpha() * (-policy_logp.detach() - self._target_entropy)
                ).mean()
                alpha_loss.backward()
                self._alpha_opt.step()

            with torch.no_grad():
                q_mean = q_all.mean(0)
                iq_reward = q_mean - y
                expert_iq_reward = iq_reward[is_expert]
                policy_mean = self._policy_mean_from_latent(expert_z.detach())
                last_stats = {
                    "dyn_loss": float(dyn_loss.item()),
                    "q_loss": float(q_loss.item()),
                    "softq_loss": float(softq_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "chi2_loss": float(chi2_loss.item()),
                    "actor_loss": float(actor_loss.item()),
                    "bc_loss": float(bc_loss.item()),
                    "alpha_loss": float(alpha_loss.item()),
                    "alpha": float(self._iqlearn_alpha().item()),
                    "iq_reward_mean": float(iq_reward.mean().item()),
                    "iq_reward_std": float(iq_reward.std(unbiased=False).item()),
                    "expert_iq_reward_mean": float(expert_iq_reward.mean().item()),
                    "expert_q_mean": float(q_mean[is_expert].mean().item()),
                    "policy_q_mean": float(q_mean[~is_expert].mean().item()) if (~is_expert).any() else 0.0,
                    "v_mean": float(current_v.mean().item()),
                    "policy_logp_mean": float(policy_logp.mean().item()),
                    "policy_expert_action_mae": float((policy_mean - expert_actions).abs().mean().item()),
                }
            if log_every > 0 and (step_idx + 1) % log_every == 0:
                print(
                    "[MPAIL2Learner] IQ-Learn pretrain "
                    f"step={step_idx + 1}/{int(steps)} "
                    f"q_loss={last_stats.get('q_loss', 0.0):.6f} "
                    f"softq={last_stats.get('softq_loss', 0.0):.6f} "
                    f"value={last_stats.get('value_loss', 0.0):.6f} "
                    f"chi2={last_stats.get('chi2_loss', 0.0):.6f} "
                    f"actor={last_stats.get('actor_loss', 0.0):.6f} "
                    f"bc={last_stats.get('bc_loss', 0.0):.6f} "
                    f"alpha={last_stats.get('alpha', 0.0):.6f} "
                    f"expert_reward={last_stats.get('expert_iq_reward_mean', 0.0):.6f} "
                    f"policy_mae={last_stats.get('policy_expert_action_mae', 0.0):.6f}",
                    flush=True,
                )

        if not was_training:
            self.planner.eval()
        print(
            "[MPAIL2Learner] IQ-Learn pretrained same MPAIL models: "
            f"steps={int(steps)} expert_batch_size={batch_size} policy_batch_size={policy_batch_size} "
            f"dyn_loss={last_stats.get('dyn_loss', 0.0):.6f} "
            f"q_loss={last_stats.get('q_loss', 0.0):.6f} "
            f"softq_loss={last_stats.get('softq_loss', 0.0):.6f} "
            f"value_loss={last_stats.get('value_loss', 0.0):.6f} "
            f"chi2_loss={last_stats.get('chi2_loss', 0.0):.6f} "
            f"actor_loss={last_stats.get('actor_loss', 0.0):.6f} "
            f"bc_loss={last_stats.get('bc_loss', 0.0):.6f} "
            f"alpha={last_stats.get('alpha', 0.0):.6f} "
            f"expert_iq_reward_mean={last_stats.get('expert_iq_reward_mean', 0.0):.6f} "
            f"policy_expert_action_mae={last_stats.get('policy_expert_action_mae', 0.0):.6f}",
            flush=True,
        )

    def _distill_iqlearn_reward(self, *, steps: int, batch_size: int):
        if self._iql_obs is None or self._iql_actions is None:
            return
        batch_size = min(int(batch_size), self._iql_obs.shape[0])
        was_training = self.planner.training
        self.planner.train()
        last_stats = {}
        for _ in range(int(steps)):
            idx = torch.randint(self._iql_obs.shape[0], (batch_size,), device=self.device)
            raw_obs = {"policy": self._iql_obs[idx]}
            raw_next_obs = {"policy": self._iql_next_obs[idx]}
            obs = self._obs_normalizer(raw_obs) if self._obs_normalizer is not None else raw_obs
            next_obs = self._obs_normalizer(raw_next_obs) if self._obs_normalizer is not None else raw_next_obs
            expert_actions = self._iql_actions[idx]
            not_done = self._iql_not_done[idx]

            with torch.no_grad():
                z = self._encoder(obs).detach()
                next_z = self._encoder(next_obs).detach()
                expert_target = self._iqlearn_reward_from_latent(z, expert_actions, next_z, not_done).detach()
                policy_actions, _ = self._sample_policy_first_action_and_logp(z)
                random_actions = torch.empty_like(expert_actions).uniform_(-1.0, 1.0)
                policy_next_z = self._dynamics(z0=z, controls=policy_actions.unsqueeze(1))[:, 1, :].detach()
                random_next_z = self._dynamics(z0=z, controls=random_actions.unsqueeze(1))[:, 1, :].detach()
                ones = torch.ones_like(not_done)
                policy_target = self._iqlearn_reward_from_latent(z, policy_actions, policy_next_z, ones).detach()
                random_target = self._iqlearn_reward_from_latent(z, random_actions, random_next_z, ones).detach()

            states = torch.cat([z, z, z], dim=0)
            next_states = torch.cat([next_z, policy_next_z, random_next_z], dim=0)
            actions = torch.cat([expert_actions, policy_actions, random_actions], dim=0)
            targets = torch.cat([expert_target, policy_target, random_target], dim=0)
            pred = self._reward(states, next_states, action=actions)
            loss = F.mse_loss(pred, targets)
            self._reward_opt.zero_grad(set_to_none=True)
            loss.backward()
            reward_grad_norm = getattr(self.reward_learner_cfg, "max_grad_norm", self.policy_learner_cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._reward.parameters(), reward_grad_norm)
            self._reward_opt.step()

            last_stats = {
                "loss": float(loss.item()),
                "target_mean": float(targets.mean().item()),
                "target_std": float(targets.std(unbiased=False).item()),
                "pred_mean": float(pred.detach().mean().item()),
                "expert_target_mean": float(expert_target.mean().item()),
                "policy_target_mean": float(policy_target.mean().item()),
                "random_target_mean": float(random_target.mean().item()),
            }
        if not was_training:
            self.planner.eval()
        print(
            "[MPAIL2Learner] Distilled IQ-Learn recovered reward into MPAIL reward model: "
            f"steps={int(steps)} batch_size={batch_size} "
            f"loss={last_stats.get('loss', 0.0):.6f} "
            f"target_mean={last_stats.get('target_mean', 0.0):.6f} "
            f"pred_mean={last_stats.get('pred_mean', 0.0):.6f}",
            flush=True,
        )

    def update_iqlearn_online(
        self,
        *,
        latent_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_latent_batch: torch.Tensor,
        replay_is_demo: torch.Tensor | None = None,
        terminal_mask: torch.Tensor | None = None,
        batch_size: int,
    ) -> dict[str, float]:
        if self._iql_obs is None or self._iql_actions is None:
            return {}

        expert_batch_size = min(int(batch_size), self._iql_obs.shape[0])
        if expert_batch_size <= 0:
            return {}

        if replay_is_demo is not None:
            replay_candidates = torch.nonzero(~replay_is_demo, as_tuple=False).flatten()
            if replay_candidates.numel() == 0:
                replay_candidates = torch.arange(latent_batch.shape[0], device=self.device)
        else:
            replay_candidates = torch.arange(latent_batch.shape[0], device=self.device)
        replay_batch_size = min(int(batch_size), int(replay_candidates.numel()))
        policy_ratio = max(float(getattr(self.cfg, "iqlearn_online_policy_batch_ratio", 1.0)), 0.0)
        policy_batch_size = max(int(round(replay_batch_size * policy_ratio)), 1) if policy_ratio > 0.0 else 0

        expert_idx = torch.randint(self._iql_obs.shape[0], (expert_batch_size,), device=self.device)
        raw_expert_obs = {"policy": self._iql_obs[expert_idx]}
        raw_expert_next_obs = {"policy": self._iql_next_obs[expert_idx]}
        expert_obs = self._obs_normalizer(raw_expert_obs) if self._obs_normalizer is not None else raw_expert_obs
        expert_next_obs = (
            self._obs_normalizer(raw_expert_next_obs)
            if self._obs_normalizer is not None
            else raw_expert_next_obs
        )

        with torch.no_grad():
            expert_z = self._encoder(expert_obs).detach()
            expert_next_z = self._encoder(expert_next_obs).detach()
            expert_actions = self._iql_actions[expert_idx].detach()
            expert_not_done = self._iql_not_done[expert_idx].detach()

        z_parts = [expert_z]
        next_z_parts = [expert_next_z]
        action_parts = [expert_actions]
        not_done_parts = [expert_not_done]
        is_expert_parts = [torch.ones(expert_batch_size, device=self.device, dtype=torch.bool)]
        is_replay_parts = [torch.zeros(expert_batch_size, device=self.device, dtype=torch.bool)]
        is_policy_parts = [torch.zeros(expert_batch_size, device=self.device, dtype=torch.bool)]

        replay_z_for_policy = None
        if replay_batch_size > 0:
            replay_idx = replay_candidates[torch.randint(replay_candidates.numel(), (replay_batch_size,), device=self.device)]
            replay_z = latent_batch[replay_idx].detach()
            replay_next_z = next_latent_batch[replay_idx].detach()
            replay_actions = action_batch[replay_idx].detach()
            if terminal_mask is not None:
                replay_not_done = (~terminal_mask[replay_idx]).to(dtype=self.dtype)
            else:
                replay_not_done = torch.ones(replay_batch_size, device=self.device, dtype=self.dtype)

            z_parts.append(replay_z)
            next_z_parts.append(replay_next_z)
            action_parts.append(replay_actions)
            not_done_parts.append(replay_not_done)
            is_expert_parts.append(torch.zeros(replay_batch_size, device=self.device, dtype=torch.bool))
            is_replay_parts.append(torch.ones(replay_batch_size, device=self.device, dtype=torch.bool))
            is_policy_parts.append(torch.zeros(replay_batch_size, device=self.device, dtype=torch.bool))
            replay_z_for_policy = replay_z

        if policy_batch_size > 0:
            if replay_z_for_policy is None:
                policy_base_z = expert_z
            else:
                policy_idx = torch.randint(replay_z_for_policy.shape[0], (policy_batch_size,), device=self.device)
                policy_base_z = replay_z_for_policy[policy_idx]
            with torch.no_grad():
                policy_actions, _ = self._sample_policy_first_action_and_logp(policy_base_z)
                policy_actions = policy_actions.detach()
                policy_next_z = self._dynamics(z0=policy_base_z, controls=policy_actions.unsqueeze(1))[:, 1, :].detach()
                policy_not_done = torch.ones(policy_batch_size, device=self.device, dtype=self.dtype)

            z_parts.append(policy_base_z.detach())
            next_z_parts.append(policy_next_z)
            action_parts.append(policy_actions)
            not_done_parts.append(policy_not_done)
            is_expert_parts.append(torch.zeros(policy_batch_size, device=self.device, dtype=torch.bool))
            is_replay_parts.append(torch.zeros(policy_batch_size, device=self.device, dtype=torch.bool))
            is_policy_parts.append(torch.ones(policy_batch_size, device=self.device, dtype=torch.bool))

        z = torch.cat(z_parts, dim=0)
        next_z = torch.cat(next_z_parts, dim=0)
        actions = torch.cat(action_parts, dim=0)
        not_done = torch.cat(not_done_parts, dim=0)
        is_expert = torch.cat(is_expert_parts, dim=0)
        is_replay = torch.cat(is_replay_parts, dim=0)
        is_policy = torch.cat(is_policy_parts, dim=0)

        alpha = max(float(getattr(self.cfg, "iqlearn_alpha", 1.0)), 1e-6)
        gamma = float(self.value_learner_cfg.gamma)
        regularize_all = bool(getattr(self.cfg, "iqlearn_online_regularize_all", True))
        learn_temp = bool(getattr(self.cfg, "iqlearn_learn_temp", False))

        with torch.no_grad():
            next_v = self._iqlearn_v_pi(next_z, target=True).detach()
            policy_actions_for_v, policy_logp_for_v = self._sample_policy_first_action_and_logp(z)
            policy_actions_for_v = policy_actions_for_v.detach()
            policy_logp_for_v = policy_logp_for_v.detach()

        q_all = self._value(z, actions, return_type="all")
        current_alpha = self._iqlearn_alpha().detach()
        current_v = self._value(z, policy_actions_for_v, return_type="min") - current_alpha * policy_logp_for_v
        y = gamma * not_done * next_v
        reward_all = q_all - y.unsqueeze(0)
        expert_reward_all = reward_all[:, is_expert]
        softq_loss = -expert_reward_all.mean()
        value_loss = (current_v - y).mean()
        chi2_source = reward_all if regularize_all else expert_reward_all
        chi2_loss = chi2_source.pow(2).mean() / (4.0 * alpha)
        q_loss = softq_loss + value_loss + chi2_loss

        self._value_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._value.parameters(), self.value_learner_cfg.max_grad_norm)
        self._value_opt.step()
        self._soft_update_target_value()

        actor_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        policy_logp = torch.zeros((), device=self.device, dtype=self.dtype)
        policy_q = torch.zeros((), device=self.device, dtype=self.dtype)
        if bool(getattr(self.cfg, "iqlearn_online_update_policy", True)):
            z_pi = z.detach()
            iq_policy_actions, policy_logp = self._sample_policy_first_action_and_logp(z_pi)
            policy_q = self._value(z_pi, iq_policy_actions, return_type="min")
            actor_loss = (self._iqlearn_alpha().detach() * policy_logp - policy_q).mean()

            self._policy_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self.policy_learner_cfg.max_grad_norm)
            self._policy_opt.step()

        alpha_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        if learn_temp and bool(getattr(self.cfg, "iqlearn_online_update_policy", True)):
            self._alpha_opt.zero_grad(set_to_none=True)
            alpha_loss = (self._iqlearn_alpha() * (-policy_logp.detach() - self._target_entropy)).mean()
            alpha_loss.backward()
            self._alpha_opt.step()

        with torch.no_grad():
            iq_reward = q_all.mean(0) - y
            stats = {
                "IQOnline/q_loss": float(q_loss.item()),
                "IQOnline/softq_loss": float(softq_loss.item()),
                "IQOnline/value_loss": float(value_loss.item()),
                "IQOnline/chi2_loss": float(chi2_loss.item()),
                "IQOnline/actor_loss": float(actor_loss.item()),
                "IQOnline/alpha_loss": float(alpha_loss.item()),
                "IQOnline/alpha": float(self._iqlearn_alpha().item()),
                "IQOnline/reward_mean": float(iq_reward.mean().item()),
                "IQOnline/expert_reward_mean": float(iq_reward[is_expert].mean().item()),
                "IQOnline/v_mean": float(current_v.mean().item()),
                "IQOnline/policy_q_mean": float(policy_q.mean().item()) if policy_q.numel() > 1 else float(policy_q.item()),
            }
            if is_replay.any():
                stats["IQOnline/replay_reward_mean"] = float(iq_reward[is_replay].mean().item())
            if is_policy.any():
                stats["IQOnline/policy_reward_mean"] = float(iq_reward[is_policy].mean().item())
        return stats

    def update_iqlearn_reward_distill_online(
        self,
        *,
        latent_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_latent_batch: torch.Tensor,
        replay_is_demo: torch.Tensor | None = None,
        terminal_mask: torch.Tensor | None = None,
        batch_size: int,
    ) -> dict[str, float]:
        if self._iql_obs is None or self._iql_actions is None:
            return {}

        expert_batch_size = min(int(batch_size), self._iql_obs.shape[0])
        if expert_batch_size <= 0:
            return {}

        if replay_is_demo is not None:
            replay_candidates = torch.nonzero(~replay_is_demo, as_tuple=False).flatten()
            if replay_candidates.numel() == 0:
                replay_candidates = torch.arange(latent_batch.shape[0], device=self.device)
        else:
            replay_candidates = torch.arange(latent_batch.shape[0], device=self.device)
        replay_batch_size = min(int(batch_size), int(replay_candidates.numel()))

        expert_idx = torch.randint(self._iql_obs.shape[0], (expert_batch_size,), device=self.device)
        raw_expert_obs = {"policy": self._iql_obs[expert_idx]}
        raw_expert_next_obs = {"policy": self._iql_next_obs[expert_idx]}
        expert_obs = self._obs_normalizer(raw_expert_obs) if self._obs_normalizer is not None else raw_expert_obs
        expert_next_obs = (
            self._obs_normalizer(raw_expert_next_obs)
            if self._obs_normalizer is not None
            else raw_expert_next_obs
        )

        with torch.no_grad():
            expert_z = self._encoder(expert_obs).detach()
            expert_next_z = self._encoder(expert_next_obs).detach()
            expert_actions = self._iql_actions[expert_idx].detach()
            expert_not_done = self._iql_not_done[expert_idx].detach()
            expert_target = self._iqlearn_reward_from_latent(
                expert_z, expert_actions, expert_next_z, expert_not_done
            ).detach()

            replay_targets = []
            replay_states = []
            replay_next_states = []
            replay_actions_list = []
            if replay_batch_size > 0:
                replay_idx = replay_candidates[
                    torch.randint(replay_candidates.numel(), (replay_batch_size,), device=self.device)
                ]
                replay_z = latent_batch[replay_idx].detach()
                replay_next_z = next_latent_batch[replay_idx].detach()
                replay_actions = action_batch[replay_idx].detach()
                if terminal_mask is not None:
                    replay_not_done = (~terminal_mask[replay_idx]).to(dtype=self.dtype)
                else:
                    replay_not_done = torch.ones(replay_batch_size, device=self.device, dtype=self.dtype)
                replay_target = self._iqlearn_reward_from_latent(
                    replay_z, replay_actions, replay_next_z, replay_not_done
                ).detach()
                replay_states.append(replay_z)
                replay_next_states.append(replay_next_z)
                replay_actions_list.append(replay_actions)
                replay_targets.append(replay_target)

                policy_actions, _ = self._sample_policy_first_action_and_logp(replay_z)
                policy_actions = policy_actions.detach()
                policy_next_z = self._dynamics(z0=replay_z, controls=policy_actions.unsqueeze(1))[:, 1, :].detach()
                policy_target = self._iqlearn_reward_from_latent(
                    replay_z,
                    policy_actions,
                    policy_next_z,
                    torch.ones(replay_batch_size, device=self.device, dtype=self.dtype),
                ).detach()
                replay_states.append(replay_z)
                replay_next_states.append(policy_next_z)
                replay_actions_list.append(policy_actions)
                replay_targets.append(policy_target)

        states = [expert_z]
        next_states = [expert_next_z]
        actions = [expert_actions]
        targets = [expert_target]
        if replay_targets:
            states.extend(replay_states)
            next_states.extend(replay_next_states)
            actions.extend(replay_actions_list)
            targets.extend(replay_targets)

        states_tensor = torch.cat(states, dim=0)
        next_states_tensor = torch.cat(next_states, dim=0)
        actions_tensor = torch.cat(actions, dim=0)
        targets_tensor = torch.cat(targets, dim=0)
        pred = self._reward(states_tensor, next_states_tensor, action=actions_tensor)
        loss = F.mse_loss(pred, targets_tensor)

        self._reward_opt.zero_grad(set_to_none=True)
        loss.backward()
        reward_grad_norm = getattr(self.reward_learner_cfg, "max_grad_norm", self.policy_learner_cfg.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self._reward.parameters(), reward_grad_norm)
        self._reward_opt.step()

        with torch.no_grad():
            stats = {
                "IQRewardDistill/online_loss": float(loss.item()),
                "IQRewardDistill/online_target_mean": float(targets_tensor.mean().item()),
                "IQRewardDistill/online_target_std": float(targets_tensor.std(unbiased=False).item()),
                "IQRewardDistill/online_pred_mean": float(pred.detach().mean().item()),
                "IQRewardDistill/online_expert_target_mean": float(expert_target.mean().item()),
            }
            if replay_targets:
                stats["IQRewardDistill/online_replay_target_mean"] = float(
                    torch.cat(replay_targets, dim=0).mean().item()
                )
        return stats

    def _pretrain_iql_on_demo_transitions(self, *, steps: int, batch_size: int):
        if self._iql_obs is None or self._iql_actions is None:
            return
        batch_size = min(int(batch_size), self._iql_obs.shape[0])
        was_training = self.planner.training
        self.planner.train()

        expectile = float(getattr(self.cfg, "iql_expectile", 0.7))
        beta = float(getattr(self.cfg, "iql_beta", 3.0))
        adv_clip = float(getattr(self.cfg, "iql_adv_clip", 100.0))
        demo_reward = float(getattr(self.cfg, "iql_demo_reward", 1.0))
        dyn_coeff = float(getattr(self.cfg, "iql_dynamics_coeff", 1.0))
        gamma = float(self.value_learner_cfg.gamma)
        last_stats = {}

        for _ in range(int(steps)):
            idx = torch.randint(self._iql_obs.shape[0], (batch_size,), device=self.device)
            raw_obs = {"policy": self._iql_obs[idx]}
            raw_next_obs = {"policy": self._iql_next_obs[idx]}
            obs = self._obs_normalizer(raw_obs) if self._obs_normalizer is not None else raw_obs
            next_obs = self._obs_normalizer(raw_next_obs) if self._obs_normalizer is not None else raw_next_obs
            actions = self._iql_actions[idx]
            not_done = self._iql_not_done[idx]

            z = self._encoder(obs)
            with torch.no_grad():
                next_z = self._encoder(next_obs).detach()

            if dyn_coeff > 0.0:
                pred_next_z = self._dynamics(z0=z, controls=actions.unsqueeze(1))[:, 1, :]
                dyn_loss = F.mse_loss(pred_next_z, next_z)
                self._dyn_opt.zero_grad(set_to_none=True)
                (dyn_coeff * dyn_loss).backward()
                torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), self.policy_learner_cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self._dynamics.parameters(), self.value_learner_cfg.max_grad_norm)
                self._dyn_opt.step()

            with torch.no_grad():
                z_value = self._encoder(obs).detach()
                next_z_value = self._encoder(next_obs).detach()
                next_pi = self._policy_mean_from_latent(next_z_value)
                next_v = self._value_target(next_z_value, next_pi, return_type="min")
                q_target = demo_reward + gamma * not_done * next_v
                demo_q_for_v = self._value(z_value, actions, return_type="avg").detach()
                pi_for_v = self._policy_mean_from_latent(z_value).detach()

            pi_q_all = self._value(z_value, pi_for_v, return_type="all")
            value_diff = demo_q_for_v.unsqueeze(0) - pi_q_all
            value_loss = self._expectile_loss(value_diff, expectile).mean()
            demo_q_all = self._value(z_value, actions, return_type="all")
            q_loss = F.mse_loss(demo_q_all, q_target.unsqueeze(0).expand_as(demo_q_all))
            value_total = q_loss + value_loss
            self._value_opt.zero_grad(set_to_none=True)
            value_total.backward()
            torch.nn.utils.clip_grad_norm_(self._value.parameters(), self.value_learner_cfg.max_grad_norm)
            self._value_opt.step()
            self._soft_update_target_value()

            z_policy = self._encoder(obs).detach()
            pi_actions = self._policy_mean_from_latent(z_policy)
            with torch.no_grad():
                demo_q = self._value(z_policy, actions, return_type="min")
                pi_q = self._value(z_policy, pi_actions.detach(), return_type="min")
                adv = demo_q - pi_q
                weights = torch.exp(beta * adv).clamp(max=adv_clip)
            policy_loss = (weights.unsqueeze(-1) * (pi_actions - actions).pow(2)).mean()
            self._policy_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self.policy_learner_cfg.max_grad_norm)
            self._policy_opt.step()

            last_stats = {
                "dyn_loss": float(dyn_loss.detach().item()) if dyn_coeff > 0.0 else 0.0,
                "q_loss": float(q_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "policy_loss": float(policy_loss.detach().item()),
                "adv_mean": float(adv.mean().item()),
                "weight_mean": float(weights.mean().item()),
                "action_mae": float((pi_actions.detach() - actions).abs().mean().item()),
            }

        if not was_training:
            self.planner.eval()
        print(
            "[MPAIL2Learner] IQL-style pretrained same MPAIL models: "
            f"steps={int(steps)} batch_size={batch_size} "
            f"dyn_loss={last_stats.get('dyn_loss', 0.0):.6f} "
            f"q_loss={last_stats.get('q_loss', 0.0):.6f} "
            f"value_loss={last_stats.get('value_loss', 0.0):.6f} "
            f"policy_loss={last_stats.get('policy_loss', 0.0):.6f} "
            f"action_mae={last_stats.get('action_mae', 0.0):.6f}",
            flush=True,
        )

    def _pretrain_policy_on_demo_actions(self, *, steps: int, batch_size: int):
        if self._bc_obs is None or self._bc_actions is None:
            return
        batch_size = min(int(batch_size), self._bc_obs.shape[0])
        was_training = self.planner.training
        self.planner.train()
        pretrain_params = [
            {"params": self._encoder.parameters(), "lr": self.dynamics_learner_cfg.opt_params["lr"]},
            {"params": self._policy.parameters(), "lr": self.policy_learner_cfg.opt_params["lr"]},
        ]
        pretrain_opt = torch.optim.Adam(pretrain_params)
        last_loss = 0.0
        last_mae = 0.0
        for _ in range(int(steps)):
            idx = torch.randint(self._bc_obs.shape[0], (batch_size,), device=self.device)
            bc_obs = {"policy": self._bc_obs[idx]}
            if self._obs_normalizer is not None:
                bc_obs = self._obs_normalizer(bc_obs)
            bc_z = self._encoder(bc_obs)
            bc_out = self._policy.model(bc_z)
            bc_mu, _bc_log_std = torch.chunk(bc_out, 2, dim=-1)
            bc_mu = torch.tanh(bc_mu.view(batch_size, self._policy.horizon, self._policy.action_dim)[:, 0])
            loss = F.mse_loss(bc_mu, self._bc_actions[idx])
            pretrain_opt.zero_grad(set_to_none=True)
            loss.backward()
            enc_grad_norm = getattr(
                self.dynamics_learner_cfg,
                "max_grad_norm",
                self.policy_learner_cfg.max_grad_norm,
            )
            torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), enc_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self.policy_learner_cfg.max_grad_norm)
            pretrain_opt.step()
            last_loss = float(loss.item())
            last_mae = float((bc_mu - self._bc_actions[idx]).abs().mean().item())
        if not was_training:
            self.planner.eval()
        print(
            f"[MPAIL2Learner] BC-pretrained policy: steps={int(steps)} "
            f"batch_size={batch_size} final_bc_loss={last_loss:.6f} final_bc_mae={last_mae:.6f}",
            flush=True,
        )

    def _bc_policy_mean_action(self, obs) -> torch.Tensor:
        return self._bc_policy_plan(obs)[:, 0]

    def _bc_policy_plan(self, obs) -> torch.Tensor:
        if self._raw_base_policy is not None and self._raw_base_policy_obs_key in obs:
            raw_obs = obs[self._raw_base_policy_obs_key]
            if self._raw_base_policy_mean is not None and self._raw_base_policy_std is not None:
                raw_obs = (raw_obs - self._raw_base_policy_mean) / (self._raw_base_policy_std + 1e-8)
                raw_obs = torch.clamp(raw_obs, min=-10.0, max=10.0)
            out = self._raw_base_policy.model(raw_obs)
            mu, _log_std = torch.chunk(out, 2, dim=-1)
            return torch.tanh(
                mu.view(mu.shape[0], self._raw_base_policy.horizon, self._raw_base_policy.action_dim)
            )
        policy_obs = obs
        if self._obs_normalizer is not None:
            policy_obs = self._obs_normalizer(policy_obs)
        z = self._encoder(policy_obs)
        out = self._policy.model(z)
        mu, _log_std = torch.chunk(out, 2, dim=-1)
        return torch.tanh(mu.view(mu.shape[0], self._policy.horizon, self._policy.action_dim))

    def _residual_adapter_action(self, obs, base_actions: torch.Tensor | None = None) -> torch.Tensor:
        if self._residual_adapter is None:
            return self._bc_policy_mean_action(obs) if base_actions is None else base_actions
        if getattr(self.cfg, "residual_adapter_use_raw_obs", False):
            z = self._residual_adapter_input(obs)
        else:
            policy_obs = obs
            if self._obs_normalizer is not None:
                policy_obs = self._obs_normalizer(policy_obs)
            with torch.no_grad():
                z = self._encoder(policy_obs)
        base = self._bc_policy_mean_action(obs) if base_actions is None else base_actions
        delta = torch.tanh(self._residual_adapter(z))
        scale = float(getattr(self.cfg, "residual_adapter_scale", 1.0))
        return (base + scale * delta).clamp(-1.0, 1.0)

    def _residual_adapter_input(self, obs) -> torch.Tensor:
        if isinstance(obs, dict):
            raw_key = getattr(self.cfg, "raw_base_policy_obs_key", "raw_policy")
            if raw_key in obs:
                adapter_obs = obs[raw_key]
            else:
                adapter_obs = obs["policy"]
        else:
            adapter_obs = obs
        if (
            getattr(self.cfg, "residual_adapter_use_raw_obs", False)
            and self._raw_base_policy_mean is not None
            and adapter_obs.shape[-1] == self._raw_base_policy_mean.shape[-1]
        ):
            adapter_obs = (adapter_obs - self._raw_base_policy_mean) / (self._raw_base_policy_std + 1e-8)
            adapter_obs = torch.clamp(adapter_obs, min=-10.0, max=10.0)
        return adapter_obs

    def online_teacher_bc_update(self, obs, actions, steps: int | None = None, batch_size: int | None = None):
        steps = int(self.cfg.online_teacher_bc_steps if steps is None else steps)
        if steps <= 0:
            return {}
        batch_size = int(self.cfg.online_teacher_bc_batch_size if batch_size is None else batch_size)
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        target = actions.detach()
        n = policy_obs.shape[0]
        batch_size = min(batch_size, n)
        was_training = self.planner.training
        self.planner.train()
        last_loss = 0.0
        last_mae = 0.0
        for _ in range(steps):
            idx = torch.randint(n, (batch_size,), device=self.device)
            pred = self._bc_policy_mean_action({"policy": policy_obs[idx]})
            loss = F.mse_loss(pred, target[idx])
            self._policy_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self.policy_learner_cfg.max_grad_norm)
            self._policy_opt.step()
            last_loss = float(loss.item())
            last_mae = float((pred - target[idx]).abs().mean().item())
        if not was_training:
            self.planner.eval()
        return {
            "OnlineTeacherBC/loss": last_loss,
            "OnlineTeacherBC/mae": last_mae,
            "OnlineTeacherBC/steps": float(steps),
        }

    def residual_adapter_update(self, obs, actions, steps: int | None = None, batch_size: int | None = None):
        if self._residual_adapter is None or self._residual_adapter_opt is None:
            return {}
        steps = int(getattr(self.cfg, "residual_adapter_steps", 0) if steps is None else steps)
        if steps <= 0:
            return {}
        batch_size = int(getattr(self.cfg, "residual_adapter_batch_size", 256) if batch_size is None else batch_size)
        if isinstance(obs, dict):
            raw_key = getattr(self.cfg, "raw_base_policy_obs_key", "raw_policy")
            source_obs = obs.get(raw_key, obs.get("policy"))
        else:
            source_obs = obs
        policy_obs = self._residual_adapter_input(obs)
        target_actions = actions.detach()
        n = policy_obs.shape[0]
        batch_size = min(batch_size, n)
        was_training = self.planner.training
        self.planner.train()
        self._residual_adapter.train()
        last_loss = 0.0
        last_action_mae = 0.0
        last_delta_mae = 0.0
        for _ in range(steps):
            idx = torch.randint(n, (batch_size,), device=self.device)
            if getattr(self.cfg, "residual_adapter_use_raw_obs", False):
                raw_key = getattr(self.cfg, "raw_base_policy_obs_key", "raw_policy")
                batch_obs = {raw_key: source_obs[idx]}
            else:
                batch_obs = {"policy": policy_obs[idx]}
            with torch.no_grad():
                base = self._bc_policy_mean_action(batch_obs)
                target_delta = (target_actions[idx] - base).clamp(-1.0, 1.0)
                if getattr(self.cfg, "residual_adapter_use_raw_obs", False):
                    z = policy_obs[idx]
                else:
                    norm_obs = batch_obs
                    if self._obs_normalizer is not None:
                        norm_obs = self._obs_normalizer(norm_obs)
                    z = self._encoder(norm_obs)
            pred_delta = torch.tanh(self._residual_adapter(z))
            scale = float(getattr(self.cfg, "residual_adapter_scale", 1.0))
            pred_action = (base + scale * pred_delta).clamp(-1.0, 1.0)
            loss = F.mse_loss(scale * pred_delta, target_delta)
            self._residual_adapter_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._residual_adapter.parameters(), self.policy_learner_cfg.max_grad_norm)
            self._residual_adapter_opt.step()
            last_loss = float(loss.item())
            last_action_mae = float((pred_action - target_actions[idx]).abs().mean().item())
            last_delta_mae = float((scale * pred_delta - target_delta).abs().mean().item())
        if not was_training:
            self.planner.eval()
            self._residual_adapter.eval()
        return {
            "ResidualAdapter/loss": last_loss,
            "ResidualAdapter/action_mae": last_action_mae,
            "ResidualAdapter/delta_mae": last_delta_mae,
            "ResidualAdapter/steps": float(steps),
        }

    def _nearest_demo_action(self, obs) -> torch.Tensor:
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        ref_obs = self._bc_obs
        query = policy_obs
        if self._obs_normalizer is not None:
            norm_obs = self._obs_normalizer({"policy": policy_obs})
            query = norm_obs["policy"] if isinstance(norm_obs, dict) else norm_obs
            norm_ref = self._obs_normalizer({"policy": ref_obs})
            ref_obs = norm_ref["policy"] if isinstance(norm_ref, dict) else norm_ref
        dist = torch.cdist(query.reshape(query.shape[0], -1), ref_obs.reshape(ref_obs.shape[0], -1))
        return self._bc_actions[dist.argmin(dim=-1)]

    def _mppi_action_library_plans(
        self,
        obs,
        base_plan: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        library_size = int(getattr(self.cfg, "mppi_action_library_size", 0))
        anchor_size = int(getattr(self.cfg, "mppi_action_library_anchor_policy_size", 0))
        residual_candidate_size = int(getattr(self.cfg, "mppi_residual_adapter_candidate_size", 0))
        if (
            library_size <= 0
            and residual_candidate_size <= 0
            and (anchor_size <= 0 or base_plan is None)
        ):
            return None

        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        ref_obs = self._bc_obs
        pieces = []
        max_candidates = self.planner.sampling.K

        if anchor_size > 0 and base_plan is not None:
            anchor_k = min(anchor_size, max_candidates)
            pieces.append(base_plan.unsqueeze(1).expand(-1, anchor_k, -1, -1))
            max_candidates -= anchor_k

        if (
            residual_candidate_size > 0
            and self._residual_adapter is not None
            and base_plan is not None
            and max_candidates > 0
        ):
            residual_k = min(residual_candidate_size, max_candidates)
            with torch.no_grad():
                base_action = base_plan[:, 0]
                residual_action = self._residual_adapter_action(obs, base_actions=base_action)
                residual_delta = residual_action - base_action
                max_scale = float(getattr(self.cfg, "mppi_residual_adapter_candidate_max_scale", 1.0))
                scales = torch.linspace(
                    max_scale / residual_k,
                    max_scale,
                    residual_k,
                    device=self.device,
                    dtype=self.dtype,
                ).view(1, residual_k, 1, 1)
                residual_plans = base_plan.unsqueeze(1).expand(-1, residual_k, -1, -1).clone()
                residual_plans[:, :, 0, :] = (
                    base_action.unsqueeze(1) + scales.squeeze(-2) * residual_delta.unsqueeze(1)
                ).clamp(-1.0, 1.0)
                pieces.append(residual_plans)
                max_candidates -= residual_k

        if library_size <= 0 or self._bc_obs is None or self._bc_actions is None or max_candidates <= 0:
            return torch.cat(pieces, dim=1).clamp(-1.0, 1.0) if pieces else None
        if policy_obs.shape[-1] != ref_obs.shape[-1]:
            return torch.cat(pieces, dim=1).clamp(-1.0, 1.0) if pieces else None

        query = policy_obs
        if self._obs_normalizer is not None:
            norm_obs = self._obs_normalizer({"policy": policy_obs})
            query = norm_obs["policy"] if isinstance(norm_obs, dict) else norm_obs
            norm_ref = self._obs_normalizer({"policy": ref_obs})
            ref_obs = norm_ref["policy"] if isinstance(norm_ref, dict) else norm_ref

        k = min(library_size, self._bc_actions.shape[0], max_candidates)
        if k <= 0:
            return torch.cat(pieces, dim=1).clamp(-1.0, 1.0) if pieces else None

        dist = torch.cdist(query.reshape(query.shape[0], -1), ref_obs.reshape(ref_obs.shape[0], -1))
        nn = dist.topk(k=k, largest=False).indices
        horizon = self.planner.sampling.T
        offsets = torch.arange(horizon, device=self.device).view(1, 1, -1)
        seq_idx = (nn.unsqueeze(-1) + offsets).clamp(max=self._bc_actions.shape[0] - 1)
        pieces.append(self._bc_actions[seq_idx])
        return torch.cat(pieces, dim=1).clamp(-1.0, 1.0)

    def _direct_demo_action(self, obs) -> torch.Tensor:
        if getattr(self.cfg, "residual_adapter_apply", False):
            return self._residual_adapter_action(obs)
        if self.cfg.demo_action_direct_mode == "knn":
            return self._nearest_demo_action(obs)
        return self._bc_policy_mean_action(obs)

    def _candidate_selector_score(
        self,
        z0: torch.Tensor,
        policy_actions: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self._candidate_selector is None:
            raise RuntimeError("Candidate selector is not initialized.")
        if candidate_actions.ndim == 2:
            features = torch.cat(
                [z0, policy_actions, candidate_actions, candidate_actions - policy_actions],
                dim=-1,
            )
            return self._candidate_selector(features).squeeze(-1)
        z_expand = z0.unsqueeze(1).expand(-1, candidate_actions.shape[1], -1)
        policy_expand = policy_actions.unsqueeze(1).expand_as(candidate_actions)
        features = torch.cat(
            [z_expand, policy_expand, candidate_actions, candidate_actions - policy_expand],
            dim=-1,
        )
        return self._candidate_selector(features).squeeze(-1)

    def _append_mppi_elite_adversarial_transitions(self):
        coeff = float(getattr(self.cfg, "mppi_elite_adversarial_reward_coeff", 0.0))
        if coeff <= 0.0:
            return
        elite_rollouts = getattr(self.planner, "_elite_rollouts", None)
        elite_actions = getattr(self.planner, "_elite_actions", None)
        elite_returns = getattr(self.planner, "_elite_returns", None)
        if elite_rollouts is None or elite_actions is None or elite_returns is None:
            return

        with torch.no_grad():
            topk = min(int(getattr(self.cfg, "mppi_elite_adversarial_topk", 16)), elite_actions.shape[1])
            if topk <= 0:
                return
            idx = elite_returns.detach().topk(k=topk, dim=1, largest=True).indices
            rollout_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, elite_rollouts.shape[-2], elite_rollouts.shape[-1]
            )
            action_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, elite_actions.shape[-2], elite_actions.shape[-1]
            )
            selected_rollouts = elite_rollouts.detach().gather(dim=1, index=rollout_idx)
            selected_actions = elite_actions.detach().gather(dim=1, index=action_idx)

            incoming = {
                "state": selected_rollouts[:, :, :-1, :].reshape(-1, selected_rollouts.shape[-1]),
                "next_state": selected_rollouts[:, :, 1:, :].reshape(-1, selected_rollouts.shape[-1]),
                "action": selected_actions.reshape(-1, selected_actions.shape[-1]),
            }

        if self._mppi_elite_adversarial_buffer is None:
            self._mppi_elite_adversarial_buffer = incoming
        else:
            for key, value in incoming.items():
                old = self._mppi_elite_adversarial_buffer.get(key)
                self._mppi_elite_adversarial_buffer[key] = value if old is None else torch.cat((old, value), dim=0)

        max_size = int(getattr(self.cfg, "mppi_elite_adversarial_buffer_size", 32768))
        if max_size > 0:
            for key, value in self._mppi_elite_adversarial_buffer.items():
                if value.shape[0] > max_size:
                    self._mppi_elite_adversarial_buffer[key] = value[-max_size:]

    def _mppi_best_candidate_action(
        self,
        policy_actions: torch.Tensor,
        obs=None,
        action_library_plans: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        score_mode = getattr(self.cfg, "mppi_best_candidate_score_mode", "sum")
        returns = self.planner._returns.detach()
        scores = returns[..., 0] if score_mode == "first" else returns.sum(dim=-1)
        raw_scores = scores
        candidate_actions = self.planner._prior_controls.detach()[:, :, 0, :]
        selector_scores = None
        if (
            getattr(self.cfg, "mppi_candidate_selector_apply", False)
            and self._candidate_selector is not None
            and obs is not None
        ):
            rank_obs = obs
            if getattr(self.planner, "_obs_normalizer", None) is not None:
                rank_obs = self.planner._obs_normalizer(rank_obs)
            z0 = self._encoder(rank_obs).detach()
            selector_scores = self._candidate_selector_score(z0, policy_actions.detach(), candidate_actions)
            scores = selector_scores
            raw_scores = scores
        candidate_delta_l2 = (candidate_actions - policy_actions.unsqueeze(1)).norm(dim=-1)
        delta_penalty = float(getattr(self.cfg, "mppi_best_candidate_delta_penalty", 0.0))
        if delta_penalty > 0.0:
            scores = scores - delta_penalty * candidate_delta_l2

        anchor_size = int(getattr(self.cfg, "mppi_action_library_anchor_policy_size", 0))
        anchor_size = min(anchor_size, scores.shape[1])
        candidate_cos_min = getattr(self.cfg, "mppi_best_candidate_policy_cos_min", None)
        candidate_cos = None
        if candidate_cos_min is not None:
            candidate_cos = F.cosine_similarity(
                candidate_actions,
                policy_actions.unsqueeze(1).expand_as(candidate_actions),
                dim=-1,
            )
            candidate_ok = candidate_cos >= float(candidate_cos_min)
            if anchor_size > 0:
                candidate_ok[:, :anchor_size] = True
            scores = scores.masked_fill(~candidate_ok, -torch.inf)
            all_masked = torch.isneginf(scores).all(dim=-1)
            if all_masked.any():
                scores = torch.where(all_masked.unsqueeze(-1), raw_scores, scores)
        best_score, best_idx = scores.max(dim=-1)
        gather_idx = best_idx.view(-1, 1, 1, 1).expand(
            -1,
            1,
            self.planner._prior_controls.shape[-2],
            self.planner._prior_controls.shape[-1],
        )
        best_plan = self.planner._prior_controls.detach().gather(dim=1, index=gather_idx).squeeze(1)
        best_actions = best_plan[:, 0]

        if anchor_size > 0 and action_library_plans is not None:
            policy_score = scores[:, :anchor_size].max(dim=-1).values
            margin = float(getattr(self.cfg, "mppi_best_candidate_policy_margin", 0.0))
            accept = best_score >= policy_score + margin
        else:
            policy_score = torch.zeros_like(best_score)
            accept = torch.ones_like(best_score, dtype=torch.bool)

        selected = torch.where(accept.unsqueeze(-1), best_actions, policy_actions)
        stats = {
            "MPPI/best_candidate_score": best_score.mean().item(),
            "MPPI/policy_anchor_score": policy_score.mean().item(),
            "MPPI/best_minus_policy_anchor": (best_score - policy_score).mean().item(),
            "MPPI/best_candidate_accept_rate": accept.float().mean().item(),
            "MPPI/best_candidate_idx_mean": best_idx.float().mean().item(),
            "MPPI/best_candidate_score_mode_first": float(score_mode == "first"),
            "MPPI/best_candidate_raw_score": raw_scores.gather(dim=1, index=best_idx.view(-1, 1)).mean().item(),
            "MPPI/best_candidate_delta_l2": candidate_delta_l2.gather(dim=1, index=best_idx.view(-1, 1)).mean().item(),
            "MPPI/best_candidate_delta_penalty": delta_penalty,
            "MPPI/best_candidate_anchor_rate": (
                (best_idx < anchor_size).float().mean().item() if anchor_size > 0 else 0.0
            ),
        }
        if candidate_cos is not None:
            stats["MPPI/best_candidate_policy_cos"] = candidate_cos.gather(
                dim=1,
                index=best_idx.view(-1, 1),
            ).mean().item()
        if selector_scores is not None:
            policy_selector_score = selector_scores[:, :anchor_size].max(dim=-1).values if anchor_size > 0 else selector_scores[:, 0]
            stats.update({
                "MPPI/selector_enabled": 1.0,
                "MPPI/selector_best_score": selector_scores.gather(dim=1, index=best_idx.view(-1, 1)).mean().item(),
                "MPPI/selector_policy_score": policy_selector_score.mean().item(),
                "MPPI/selector_best_minus_policy": (
                    selector_scores.gather(dim=1, index=best_idx.view(-1, 1)).squeeze(-1) - policy_selector_score
                ).mean().item(),
            })
        return selected, stats, accept

    def _stateless_mppi_action(self, obs) -> torch.Tensor | None:
        """Run MPPI for a fixed-size diagnostic batch without advancing rollout state."""
        batch = next(iter(obs.values())).shape[0]
        if batch != self.planner.num_envs:
            return None

        saved_current_obs = self.planner._current_obs
        saved_opt_controls = self.planner._opt_controls.detach().clone()
        saved_prev_z = self.planner._prev_z.detach().clone()
        saved_last_actions = self.planner._last_actions.detach().clone()
        saved_iter_mean = self.planner.sampling._iter_mean.detach().clone()
        saved_iter_std = self.planner.sampling._iter_std.detach().clone()

        try:
            base_plan = None
            if getattr(self.cfg, "mppi_residual_base_policy", False):
                base_plan = self._bc_policy_plan(obs)
            action_library_plans = self._mppi_action_library_plans(obs, base_plan=base_plan)
            return self.planner.act(
                obs,
                use_prev_opt=False,
                deterministic=True,
                base_plan=base_plan,
                action_library_plans=action_library_plans,
            ).detach()
        finally:
            self.planner._current_obs = saved_current_obs
            self.planner._opt_controls.copy_(saved_opt_controls)
            self.planner._prev_z.copy_(saved_prev_z)
            self.planner._last_actions.copy_(saved_last_actions)
            self.planner.sampling._iter_mean.copy_(saved_iter_mean)
            self.planner.sampling._iter_std.copy_(saved_iter_std)

    def _demo_action_dynamics_loss(self, bc_idx: torch.Tensor):
        bc_obs = {"policy": self._bc_obs[bc_idx]}
        bc_next_obs = {"policy": self._bc_next_obs[bc_idx]}
        if self._obs_normalizer is not None:
            bc_obs = self._obs_normalizer(bc_obs)
            bc_next_obs = self._obs_normalizer(bc_next_obs)

        bc_z = self._encoder(bc_obs)
        with torch.no_grad():
            bc_next_z = self._encoder(bc_next_obs)
        target_next_z = bc_next_z.detach()
        teacher_actions = self._bc_actions[bc_idx]
        teacher_next_z = self._dynamics(z0=bc_z, controls=teacher_actions.unsqueeze(1))[:, 1, :]
        teacher_dist = (teacher_next_z - target_next_z).pow(2).mean(dim=-1)
        dyn_loss = teacher_dist.mean()
        aux_total, aux_stats = self._task_dynamics_aux_loss(
            bc_obs["policy"],
            teacher_next_z,
            target_next_z,
        )

        zero_actions = torch.zeros_like(teacher_actions)
        rand_actions = torch.empty_like(teacher_actions).uniform_(-1.0, 1.0)
        zero_next_z = self._dynamics(z0=bc_z, controls=zero_actions.unsqueeze(1))[:, 1, :]
        rand_next_z = self._dynamics(z0=bc_z, controls=rand_actions.unsqueeze(1))[:, 1, :]
        zero_dist = (zero_next_z - target_next_z).pow(2).mean(dim=-1)
        rand_dist = (rand_next_z - target_next_z).pow(2).mean(dim=-1)
        margin = self.cfg.demo_action_rank_margin
        rank_loss = (
            F.relu(margin + teacher_dist - zero_dist).mean()
            + F.relu(margin + teacher_dist - rand_dist).mean()
        )
        total = self.cfg.demo_action_dyn_coeff * dyn_loss + self.cfg.demo_action_rank_coeff * rank_loss + aux_total
        stats = {
            "dyn_loss": dyn_loss.item(),
            "rank_loss": rank_loss.item(),
            "teacher_dist": teacher_dist.mean().item(),
            "zero_dist": zero_dist.mean().item(),
            "rand_dist": rand_dist.mean().item(),
            "margin_zero": (zero_dist - teacher_dist).mean().item(),
            "margin_rand": (rand_dist - teacher_dist).mean().item(),
            "task_obs_loss": aux_stats["task_obs_loss"],
            "progress_loss": aux_stats["progress_loss"],
            "progress_sign_loss": aux_stats["progress_sign_loss"],
            "progress_over_loss": aux_stats["progress_over_loss"],
            "progress_sign_acc_train": aux_stats["progress_sign_acc_train"],
        }
        return total, stats

    def _demo_action_multistep_dynamics_loss(self, bc_idx: torch.Tensor):
        coeff = float(getattr(self.cfg, "demo_action_multistep_dyn_coeff", 0.0))
        horizon = int(getattr(self.cfg, "demo_action_multistep_dyn_horizon", 3))
        if coeff <= 0.0 or horizon <= 1 or self._bc_obs is None or self._bc_actions is None:
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
            return zero, {"multistep_loss": 0.0, "multistep_final_dist": 0.0}

        max_start = self._bc_obs.shape[0] - horizon - 1
        if max_start <= 0:
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
            return zero, {"multistep_loss": 0.0, "multistep_final_dist": 0.0}

        start_idx = bc_idx.clamp(max=max_start)
        offsets = torch.arange(horizon, device=self.device).view(1, -1)
        action_idx = start_idx.view(-1, 1) + offsets
        target_idx = start_idx.view(-1, 1) + offsets + 1

        rollout_obs = {"policy": self._bc_obs[start_idx]}
        target_obs = {"policy": self._bc_obs[target_idx.reshape(-1)]}
        if self._obs_normalizer is not None:
            rollout_obs = self._obs_normalizer(rollout_obs)
            target_obs = self._obs_normalizer(target_obs)

        z0 = self._encoder(rollout_obs)
        actions = self._bc_actions[action_idx]
        pred_z = self._dynamics(z0=z0, controls=actions)[:, 1:, :]
        with torch.no_grad():
            target_z = self._encoder(target_obs).view(start_idx.shape[0], horizon, -1).detach()

        step_dist = (pred_z - target_z).pow(2).mean(dim=-1)
        multistep_loss = step_dist.mean()
        total = coeff * multistep_loss
        stats = {
            "multistep_loss": multistep_loss.item(),
            "multistep_final_dist": step_dist[:, -1].mean().item(),
        }
        return total, stats

    def _denormalize_policy_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        if self._obs_normalizer is None:
            return obs
        return self._obs_normalizer.inverse({"policy": obs})["policy"]

    def _task_dynamics_aux_loss(
        self,
        current_obs: torch.Tensor,
        pred_next_obs: torch.Tensor,
        target_next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        zero = torch.zeros((), device=self.device, dtype=self.dtype)
        task_coeff = float(getattr(self.cfg, "dyn_task_obs_coeff", 0.0))
        progress_coeff = float(getattr(self.cfg, "dyn_progress_coeff", 0.0))
        progress_sign_coeff = float(getattr(self.cfg, "dyn_progress_sign_coeff", 0.0))
        progress_over_coeff = float(getattr(self.cfg, "dyn_progress_over_coeff", 0.0))
        stats = {
            "task_obs_loss": 0.0,
            "progress_loss": 0.0,
            "progress_sign_loss": 0.0,
            "progress_over_loss": 0.0,
            "progress_sign_acc_train": 0.0,
        }
        if (
            task_coeff <= 0.0
            and progress_coeff <= 0.0
            and progress_sign_coeff <= 0.0
            and progress_over_coeff <= 0.0
        ):
            return zero, stats
        if not getattr(self.cfg, "identity_encoder", False):
            return zero, stats
        if current_obs.shape[-1] < 60 or pred_next_obs.shape[-1] < 60 or target_next_obs.shape[-1] < 60:
            return zero, stats

        current_raw = self._denormalize_policy_obs_tensor(current_obs)
        pred_raw = self._denormalize_policy_obs_tensor(pred_next_obs)
        target_raw = self._denormalize_policy_obs_tensor(target_next_obs)

        total = zero
        if task_coeff > 0.0:
            weighted_losses = []
            for sl, weight in (
                (slice(0, 6), 4.0),    # peg in hole pose
                (slice(6, 12), 2.0),   # peg in hole delta
                (slice(30, 36), 2.0),  # ee to peg
                (slice(36, 42), 2.0),  # ee to hole
                (slice(48, 54), 1.0),  # ee delta
                (slice(54, 60), 2.0),  # peg delta
            ):
                weighted_losses.append(weight * F.mse_loss(pred_raw[..., sl], target_raw[..., sl]))
            task_obs_loss = torch.stack(weighted_losses).sum() / 13.0
            total = total + task_coeff * task_obs_loss
            stats["task_obs_loss"] = float(task_obs_loss.detach().item())

        current_dist = current_raw[..., 0:3].norm(dim=-1)
        pred_dist = pred_raw[..., 0:3].norm(dim=-1)
        target_dist = target_raw[..., 0:3].norm(dim=-1)
        pred_progress = current_dist - pred_dist
        target_progress = current_dist - target_dist

        if progress_coeff > 0.0:
            progress_loss = F.mse_loss(pred_progress, target_progress)
            total = total + progress_coeff * progress_loss
            stats["progress_loss"] = float(progress_loss.detach().item())

        if progress_over_coeff > 0.0:
            progress_over_loss = F.relu(pred_progress - target_progress).pow(2).mean()
            total = total + progress_over_coeff * progress_over_loss
            stats["progress_over_loss"] = float(progress_over_loss.detach().item())

        target_sign = torch.where(target_progress >= 0.0, 1.0, -1.0)
        pred_sign = torch.where(pred_progress >= 0.0, 1.0, -1.0)
        sign_acc = (pred_sign == target_sign).to(dtype=self.dtype).mean()
        stats["progress_sign_acc_train"] = float(sign_acc.detach().item())
        if progress_sign_coeff > 0.0:
            progress_scale = 0.002
            sign_loss = F.softplus(-target_sign * pred_progress / progress_scale).mean()
            total = total + progress_sign_coeff * sign_loss
            stats["progress_sign_loss"] = float(sign_loss.detach().item())

        return total, stats

    def _counterfactual_progress_conservative_loss(
        self,
        current_obs: torch.Tensor,
        verified_next_obs: torch.Tensor,
        counterfactual_next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        coeff = float(getattr(self.cfg, "dyn_counterfactual_progress_coeff", 0.0))
        if coeff <= 0.0 or not getattr(self.cfg, "identity_encoder", False):
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
            return zero, 0.0
        if current_obs.shape[-1] < 3 or verified_next_obs.shape[-1] < 3 or counterfactual_next_obs.shape[-1] < 3:
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
            return zero, 0.0

        current_raw = self._denormalize_policy_obs_tensor(current_obs)
        verified_raw = self._denormalize_policy_obs_tensor(verified_next_obs)
        counter_raw = self._denormalize_policy_obs_tensor(counterfactual_next_obs)
        current_dist = current_raw[..., 0:3].norm(dim=-1)
        verified_progress = current_dist - verified_raw[..., 0:3].norm(dim=-1)
        counter_progress = current_dist - counter_raw[..., 0:3].norm(dim=-1)
        loss = F.relu(counter_progress - verified_progress).pow(2).mean()
        return coeff * loss, float(loss.detach().item())

    def _demo_action_reward_rank_loss(self, bc_idx: torch.Tensor):
        bc_obs_raw = {"policy": self._bc_obs[bc_idx]}
        bc_obs = bc_obs_raw
        if self._obs_normalizer is not None:
            bc_obs = self._obs_normalizer(bc_obs)

        with torch.no_grad():
            bc_z = self._encoder(bc_obs).detach()
            teacher_actions = self._bc_actions[bc_idx].detach()
            policy_actions = self._bc_policy_mean_action(bc_obs_raw).detach()
            zero_actions = torch.zeros_like(teacher_actions)
            rand_actions = torch.empty_like(teacher_actions).uniform_(-1.0, 1.0)

            def predict_next(action: torch.Tensor) -> torch.Tensor:
                return self._dynamics(z0=bc_z, controls=action.unsqueeze(1))[:, 1, :].detach()

            teacher_next_z = predict_next(teacher_actions)
            policy_next_z = predict_next(policy_actions)
            zero_next_z = predict_next(zero_actions)
            rand_next_z = predict_next(rand_actions)

        teacher_reward = self._reward(bc_z, teacher_next_z, action=teacher_actions)
        policy_reward = self._reward(bc_z, policy_next_z, action=policy_actions)
        zero_reward = self._reward(bc_z, zero_next_z, action=zero_actions)
        rand_reward = self._reward(bc_z, rand_next_z, action=rand_actions)

        margin = float(getattr(self.cfg, "demo_action_reward_rank_margin", 0.02))
        policy_weight = float(getattr(self.cfg, "demo_action_reward_rank_policy_weight", 1.0))
        mppi_weight = float(getattr(self.cfg, "demo_action_reward_rank_mppi_weight", 0.0))
        blended_weight = float(getattr(self.cfg, "demo_action_reward_rank_blended_weight", 0.0))
        policy_rank_loss = F.relu(margin - (teacher_reward - policy_reward)).mean()
        rank_loss = (
            policy_weight * policy_rank_loss
            + F.relu(margin - (teacher_reward - zero_reward)).mean()
            + F.relu(margin - (teacher_reward - rand_reward)).mean()
        )
        mppi_rank_stats = {}
        if mppi_weight > 0.0 or blended_weight > 0.0:
            mppi_batch = min(int(self.planner.num_envs), int(bc_idx.shape[0]))
            if mppi_batch == int(self.planner.num_envs):
                sub_raw = {"policy": self._bc_obs[bc_idx[:mppi_batch]]}
                sub_obs = self._obs_normalizer(sub_raw) if self._obs_normalizer is not None else sub_raw
                with torch.no_grad():
                    sub_z = self._encoder(sub_obs).detach()
                    sub_teacher = self._bc_actions[bc_idx[:mppi_batch]].detach()
                    sub_policy = self._bc_policy_mean_action(sub_raw).detach()
                    sub_mppi = self._stateless_mppi_action(sub_raw)
                    if sub_mppi is not None:
                        blend = getattr(self.cfg, "demo_action_reward_rank_mppi_blend", None)
                        blend = float(getattr(self.cfg, "mppi_policy_blend", 1.0) if blend is None else blend)
                        sub_blended = (sub_policy + blend * (sub_mppi - sub_policy)).clamp(-1.0, 1.0)
                        sub_teacher_next = self._dynamics(z0=sub_z, controls=sub_teacher.unsqueeze(1))[:, 1, :].detach()
                        sub_policy_next = self._dynamics(z0=sub_z, controls=sub_policy.unsqueeze(1))[:, 1, :].detach()
                        sub_mppi_next = self._dynamics(z0=sub_z, controls=sub_mppi.unsqueeze(1))[:, 1, :].detach()
                        sub_blended_next = self._dynamics(z0=sub_z, controls=sub_blended.unsqueeze(1))[:, 1, :].detach()

                if sub_mppi is not None:
                    sub_teacher_reward = self._reward(sub_z, sub_teacher_next, action=sub_teacher)
                    sub_policy_reward = self._reward(sub_z, sub_policy_next, action=sub_policy)
                    sub_mppi_reward = self._reward(sub_z, sub_mppi_next, action=sub_mppi)
                    sub_blended_reward = self._reward(sub_z, sub_blended_next, action=sub_blended)
                    mppi_teacher_loss = F.relu(margin - (sub_teacher_reward - sub_mppi_reward)).mean()
                    mppi_policy_loss = F.relu(margin - (sub_policy_reward - sub_mppi_reward)).mean()
                    blended_teacher_loss = F.relu(margin - (sub_teacher_reward - sub_blended_reward)).mean()
                    blended_policy_loss = F.relu(margin - (sub_policy_reward - sub_blended_reward)).mean()
                    rank_loss = rank_loss + mppi_weight * (mppi_teacher_loss + mppi_policy_loss)
                    rank_loss = rank_loss + blended_weight * (blended_teacher_loss + blended_policy_loss)
                    mppi_rank_stats = {
                        "mppi_reward": sub_mppi_reward.mean().item(),
                        "blended_reward": sub_blended_reward.mean().item(),
                        "teacher_minus_mppi": (sub_teacher_reward - sub_mppi_reward).mean().item(),
                        "policy_minus_mppi": (sub_policy_reward - sub_mppi_reward).mean().item(),
                        "teacher_minus_blended": (sub_teacher_reward - sub_blended_reward).mean().item(),
                        "policy_minus_blended": (sub_policy_reward - sub_blended_reward).mean().item(),
                        "teacher_gt_mppi": (sub_teacher_reward > sub_mppi_reward).float().mean().item(),
                        "policy_gt_mppi": (sub_policy_reward > sub_mppi_reward).float().mean().item(),
                        "teacher_gt_blended": (sub_teacher_reward > sub_blended_reward).float().mean().item(),
                        "policy_gt_blended": (sub_policy_reward > sub_blended_reward).float().mean().item(),
                        "mppi_teacher_rank_loss": mppi_teacher_loss.item(),
                        "mppi_policy_rank_loss": mppi_policy_loss.item(),
                        "blended_teacher_rank_loss": blended_teacher_loss.item(),
                        "blended_policy_rank_loss": blended_policy_loss.item(),
                    }
        total = float(getattr(self.cfg, "demo_action_reward_rank_coeff", 0.0)) * rank_loss
        stats = {
            "rank_loss": rank_loss.item(),
            "policy_rank_loss": policy_rank_loss.item(),
            "policy_rank_weight": policy_weight,
            "mppi_rank_weight": mppi_weight,
            "blended_rank_weight": blended_weight,
            "teacher_reward": teacher_reward.mean().item(),
            "policy_reward": policy_reward.mean().item(),
            "zero_reward": zero_reward.mean().item(),
            "rand_reward": rand_reward.mean().item(),
            "teacher_minus_policy": (teacher_reward - policy_reward).mean().item(),
            "teacher_minus_zero": (teacher_reward - zero_reward).mean().item(),
            "teacher_minus_rand": (teacher_reward - rand_reward).mean().item(),
            "teacher_gt_policy": (teacher_reward > policy_reward).float().mean().item(),
        }
        stats.update(mppi_rank_stats)
        return total, stats

    def online_mppi_reward_rank_update(
        self,
        obs: torch.Tensor,
        teacher_actions: torch.Tensor,
        policy_actions: torch.Tensor,
        mppi_actions: torch.Tensor,
        blended_actions: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        coeff = float(getattr(self.cfg, "online_mppi_reward_rank_coeff", 0.0))
        steps = int(getattr(self.cfg, "online_mppi_reward_rank_steps", 1))
        if coeff <= 0.0 or steps <= 0 or obs.numel() == 0:
            return {}

        incoming = {
            "obs": obs.detach(),
            "teacher": teacher_actions.detach(),
            "policy": policy_actions.detach(),
            "mppi": mppi_actions.detach(),
        }
        if blended_actions is not None:
            incoming["blended"] = blended_actions.detach()

        if self._online_mppi_rank_buffer is None:
            self._online_mppi_rank_buffer = incoming
        else:
            for key, value in incoming.items():
                old = self._online_mppi_rank_buffer.get(key)
                self._online_mppi_rank_buffer[key] = value if old is None else torch.cat((old, value), dim=0)
            for key in list(self._online_mppi_rank_buffer):
                if key not in incoming:
                    self._online_mppi_rank_buffer.pop(key)

        max_size = int(getattr(self.cfg, "online_mppi_reward_rank_buffer_size", 8192))
        if max_size > 0:
            for key, value in self._online_mppi_rank_buffer.items():
                if value.shape[0] > max_size:
                    self._online_mppi_rank_buffer[key] = value[-max_size:]

        buffer = self._online_mppi_rank_buffer
        n = buffer["obs"].shape[0]
        batch_size = min(int(getattr(self.cfg, "online_mppi_reward_rank_batch_size", 256)), n)
        margin = float(getattr(self.cfg, "online_mppi_reward_rank_margin", 0.02))
        mppi_weight = float(getattr(self.cfg, "online_mppi_reward_rank_mppi_weight", 1.0))
        policy_weight = float(getattr(self.cfg, "online_mppi_reward_rank_policy_weight", 1.0))
        blended_weight = float(getattr(self.cfg, "online_mppi_reward_rank_blended_weight", 0.0))
        next_state_weight = float(getattr(self.cfg, "online_mppi_next_state_rank_weight", 0.0))

        last_stats = {}
        for _ in range(steps):
            idx = torch.randint(n, (batch_size,), device=self.device)
            raw_obs = {"policy": buffer["obs"][idx]}
            rank_obs = self._obs_normalizer(raw_obs) if self._obs_normalizer is not None else raw_obs
            with torch.no_grad():
                z0 = self._encoder(rank_obs).detach()
                teacher = buffer["teacher"][idx].detach()
                policy = buffer["policy"][idx].detach()
                mppi = buffer["mppi"][idx].detach()
                blended = buffer.get("blended")
                blended = None if blended is None else blended[idx].detach()

                def predict_next(action: torch.Tensor) -> torch.Tensor:
                    return self._dynamics(z0=z0, controls=action.unsqueeze(1))[:, 1, :].detach()

                teacher_next = predict_next(teacher)
                policy_next = predict_next(policy)
                mppi_next = predict_next(mppi)
                blended_next = None if blended is None else predict_next(blended)

            teacher_reward = self._reward(z0, teacher_next, action=teacher)
            policy_reward = self._reward(z0, policy_next, action=policy)
            mppi_reward = self._reward(z0, mppi_next, action=mppi)

            policy_rank_loss = F.relu(margin - (teacher_reward - policy_reward)).mean()
            mppi_teacher_loss = F.relu(margin - (teacher_reward - mppi_reward)).mean()
            mppi_policy_loss = F.relu(margin - (policy_reward - mppi_reward)).mean()
            rank_loss = policy_weight * policy_rank_loss + mppi_weight * (mppi_teacher_loss + mppi_policy_loss)

            next_teacher_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            next_policy_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            teacher_next_same_action_reward = None
            policy_next_same_action_reward = None
            mppi_next_same_action_reward = None
            if next_state_weight > 0.0:
                same_action = policy
                teacher_next_same_action_reward = self._reward(z0, teacher_next, action=same_action)
                policy_next_same_action_reward = self._reward(z0, policy_next, action=same_action)
                mppi_next_same_action_reward = self._reward(z0, mppi_next, action=same_action)
                next_teacher_loss = F.relu(
                    margin - (teacher_next_same_action_reward - mppi_next_same_action_reward)
                ).mean()
                next_policy_loss = F.relu(
                    margin - (policy_next_same_action_reward - mppi_next_same_action_reward)
                ).mean()
                rank_loss = rank_loss + next_state_weight * (next_teacher_loss + next_policy_loss)

            blended_teacher_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            blended_policy_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            if blended is not None and blended_weight > 0.0:
                blended_reward = self._reward(z0, blended_next, action=blended)
                blended_teacher_loss = F.relu(margin - (teacher_reward - blended_reward)).mean()
                blended_policy_loss = F.relu(margin - (policy_reward - blended_reward)).mean()
                rank_loss = rank_loss + blended_weight * (blended_teacher_loss + blended_policy_loss)
            else:
                blended_reward = None

            loss = coeff * rank_loss
            self._reward_opt.zero_grad(set_to_none=True)
            loss.backward()
            self._reward_opt.step()

            with torch.no_grad():
                last_stats = {
                    "OnlineMPPIRank/loss": loss.item(),
                    "OnlineMPPIRank/rank_loss": rank_loss.item(),
                    "OnlineMPPIRank/buffer_size": float(n),
                    "OnlineMPPIRank/teacher_reward": teacher_reward.mean().item(),
                    "OnlineMPPIRank/policy_reward": policy_reward.mean().item(),
                    "OnlineMPPIRank/mppi_reward": mppi_reward.mean().item(),
                    "OnlineMPPIRank/teacher_minus_mppi": (teacher_reward - mppi_reward).mean().item(),
                    "OnlineMPPIRank/teacher_minus_policy": (teacher_reward - policy_reward).mean().item(),
                    "OnlineMPPIRank/policy_minus_mppi": (policy_reward - mppi_reward).mean().item(),
                    "OnlineMPPIRank/teacher_gt_mppi": (teacher_reward > mppi_reward).float().mean().item(),
                    "OnlineMPPIRank/teacher_gt_policy": (teacher_reward > policy_reward).float().mean().item(),
                    "OnlineMPPIRank/policy_gt_mppi": (policy_reward > mppi_reward).float().mean().item(),
                    "OnlineMPPIRank/policy_rank_loss": policy_rank_loss.item(),
                    "OnlineMPPIRank/mppi_teacher_rank_loss": mppi_teacher_loss.item(),
                    "OnlineMPPIRank/mppi_policy_rank_loss": mppi_policy_loss.item(),
                    "OnlineMPPIRank/next_state_weight": next_state_weight,
                    "OnlineMPPIRank/next_teacher_rank_loss": next_teacher_loss.item(),
                    "OnlineMPPIRank/next_policy_rank_loss": next_policy_loss.item(),
                    "OnlineMPPIRank/steps": float(steps),
                }
                if mppi_next_same_action_reward is not None:
                    last_stats.update({
                        "OnlineMPPIRank/next_teacher_reward": teacher_next_same_action_reward.mean().item(),
                        "OnlineMPPIRank/next_policy_reward": policy_next_same_action_reward.mean().item(),
                        "OnlineMPPIRank/next_mppi_reward": mppi_next_same_action_reward.mean().item(),
                        "OnlineMPPIRank/next_teacher_minus_mppi": (
                            teacher_next_same_action_reward - mppi_next_same_action_reward
                        ).mean().item(),
                        "OnlineMPPIRank/next_policy_minus_mppi": (
                            policy_next_same_action_reward - mppi_next_same_action_reward
                        ).mean().item(),
                        "OnlineMPPIRank/next_teacher_gt_mppi": (
                            teacher_next_same_action_reward > mppi_next_same_action_reward
                        ).float().mean().item(),
                        "OnlineMPPIRank/next_policy_gt_mppi": (
                            policy_next_same_action_reward > mppi_next_same_action_reward
                        ).float().mean().item(),
                    })
                if blended_reward is not None:
                    last_stats.update({
                        "OnlineMPPIRank/blended_reward": blended_reward.mean().item(),
                        "OnlineMPPIRank/teacher_minus_blended": (
                            teacher_reward - blended_reward
                        ).mean().item(),
                        "OnlineMPPIRank/policy_minus_blended": (
                            policy_reward - blended_reward
                        ).mean().item(),
                        "OnlineMPPIRank/teacher_gt_blended": (
                            teacher_reward > blended_reward
                        ).float().mean().item(),
                        "OnlineMPPIRank/policy_gt_blended": (
                            policy_reward > blended_reward
                        ).float().mean().item(),
                        "OnlineMPPIRank/blended_teacher_rank_loss": blended_teacher_loss.item(),
                        "OnlineMPPIRank/blended_policy_rank_loss": blended_policy_loss.item(),
                    })
        return last_stats

    def online_mppi_outcome_rank_update(
        self,
        obs: torch.Tensor,
        policy_actions: torch.Tensor,
        mppi_actions: torch.Tensor,
        blended_actions: torch.Tensor,
        good_mask: torch.Tensor,
        bad_mask: torch.Tensor,
    ) -> Dict[str, float]:
        coeff = float(getattr(self.cfg, "online_mppi_outcome_rank_coeff", 0.0))
        steps = int(getattr(self.cfg, "online_mppi_outcome_rank_steps", 1))
        if coeff <= 0.0 or steps <= 0 or obs.numel() == 0:
            return {}
        train_selector = self._candidate_selector is not None and getattr(
            self.cfg, "mppi_candidate_selector_enabled", False
        )

        usable = (good_mask | bad_mask).detach().bool()
        if usable.numel() == 0 or not usable.any():
            return {"OnlineMPPIOutcomeRank/usable_rate": 0.0}

        incoming = {
            "obs": obs[usable].detach(),
            "policy": policy_actions[usable].detach(),
            "mppi": mppi_actions[usable].detach(),
            "blended": blended_actions[usable].detach(),
            "good": good_mask[usable].detach().bool(),
            "bad": bad_mask[usable].detach().bool(),
        }

        if self._online_mppi_outcome_rank_buffer is None:
            self._online_mppi_outcome_rank_buffer = incoming
        else:
            for key, value in incoming.items():
                old = self._online_mppi_outcome_rank_buffer.get(key)
                self._online_mppi_outcome_rank_buffer[key] = value if old is None else torch.cat((old, value), dim=0)

        max_size = int(getattr(self.cfg, "online_mppi_outcome_rank_buffer_size", 8192))
        if max_size > 0:
            for key, value in self._online_mppi_outcome_rank_buffer.items():
                if value.shape[0] > max_size:
                    self._online_mppi_outcome_rank_buffer[key] = value[-max_size:]

        buffer = self._online_mppi_outcome_rank_buffer
        n = buffer["obs"].shape[0]
        batch_size = min(int(getattr(self.cfg, "online_mppi_outcome_rank_batch_size", 256)), n)
        margin = float(getattr(self.cfg, "online_mppi_outcome_rank_margin", 0.02))
        good_weight = float(getattr(self.cfg, "online_mppi_outcome_good_weight", 1.0))
        bad_weight = float(getattr(self.cfg, "online_mppi_outcome_bad_weight", 1.0))

        last_stats = {}
        for _ in range(steps):
            idx = torch.randint(n, (batch_size,), device=self.device)
            raw_obs = {"policy": buffer["obs"][idx]}
            rank_obs = self._obs_normalizer(raw_obs) if self._obs_normalizer is not None else raw_obs
            with torch.no_grad():
                z0 = self._encoder(rank_obs).detach()
                policy = buffer["policy"][idx].detach()
                mppi = buffer["mppi"][idx].detach()
                blended = buffer["blended"][idx].detach()
                good = buffer["good"][idx].detach()
                bad = buffer["bad"][idx].detach()

            if train_selector:
                policy_score = self._candidate_selector_score(z0, policy, policy)
                mppi_score = self._candidate_selector_score(z0, policy, mppi)
                blended_score = self._candidate_selector_score(z0, policy, blended)
            else:
                with torch.no_grad():
                    def predict_next(action: torch.Tensor) -> torch.Tensor:
                        return self._dynamics(z0=z0, controls=action.unsqueeze(1))[:, 1, :].detach()

                    policy_next = predict_next(policy)
                    mppi_next = predict_next(mppi)
                    blended_next = predict_next(blended)
                policy_score = self._reward(z0, policy_next, action=policy)
                mppi_score = self._reward(z0, mppi_next, action=mppi)
                blended_score = self._reward(z0, blended_next, action=blended)

            good_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            if good.any():
                good_loss = (
                    F.relu(margin - (mppi_score[good] - policy_score[good])).mean()
                    + F.relu(margin - (blended_score[good] - policy_score[good])).mean()
                )
            bad_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            if bad.any():
                bad_loss = (
                    F.relu(margin - (policy_score[bad] - mppi_score[bad])).mean()
                    + F.relu(margin - (policy_score[bad] - blended_score[bad])).mean()
                )
            rank_loss = good_weight * good_loss + bad_weight * bad_loss
            loss = coeff * rank_loss

            opt = self._candidate_selector_opt if train_selector else self._reward_opt
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                last_stats = {
                    "OnlineMPPIOutcomeRank/loss": loss.item(),
                    "OnlineMPPIOutcomeRank/rank_loss": rank_loss.item(),
                    "OnlineMPPIOutcomeRank/good_loss": good_loss.item(),
                    "OnlineMPPIOutcomeRank/bad_loss": bad_loss.item(),
                    "OnlineMPPIOutcomeRank/buffer_size": float(n),
                    "OnlineMPPIOutcomeRank/good_rate": buffer["good"].float().mean().item(),
                    "OnlineMPPIOutcomeRank/bad_rate": buffer["bad"].float().mean().item(),
                    "OnlineMPPIOutcomeRank/train_selector": float(train_selector),
                    "OnlineMPPIOutcomeRank/policy_reward": policy_score.mean().item(),
                    "OnlineMPPIOutcomeRank/mppi_reward": mppi_score.mean().item(),
                    "OnlineMPPIOutcomeRank/blended_reward": blended_score.mean().item(),
                    "OnlineMPPIOutcomeRank/policy_minus_mppi": (policy_score - mppi_score).mean().item(),
                    "OnlineMPPIOutcomeRank/blended_minus_policy": (blended_score - policy_score).mean().item(),
                    "OnlineMPPIOutcomeRank/steps": float(steps),
                }
        last_stats["OnlineMPPIOutcomeRank/usable_rate"] = usable.float().mean().item()
        return last_stats

    def online_mppi_candidate_rank_update(
        self,
        obs: torch.Tensor,
        policy_actions: torch.Tensor,
        good_actions: torch.Tensor,
        bad_actions: torch.Tensor,
        good_mask: torch.Tensor,
    ) -> Dict[str, float]:
        coeff = float(getattr(self.cfg, "online_mppi_candidate_rank_coeff", 0.0))
        steps = int(getattr(self.cfg, "online_mppi_candidate_rank_steps", 1))
        if coeff <= 0.0 or steps <= 0 or obs.numel() == 0:
            return {}
        train_selector = self._candidate_selector is not None and getattr(
            self.cfg, "mppi_candidate_selector_enabled", False
        )

        good_mask = good_mask.detach().bool()
        incoming = {
            "obs": obs.detach(),
            "policy": policy_actions.detach(),
            "good": good_actions.detach(),
            "bad": bad_actions.detach(),
            "good_mask": good_mask,
        }

        if self._online_mppi_candidate_rank_buffer is None:
            self._online_mppi_candidate_rank_buffer = incoming
        else:
            for key, value in incoming.items():
                old = self._online_mppi_candidate_rank_buffer.get(key)
                self._online_mppi_candidate_rank_buffer[key] = value if old is None else torch.cat((old, value), dim=0)

        max_size = int(getattr(self.cfg, "online_mppi_candidate_rank_buffer_size", 8192))
        if max_size > 0:
            for key, value in self._online_mppi_candidate_rank_buffer.items():
                if value.shape[0] > max_size:
                    self._online_mppi_candidate_rank_buffer[key] = value[-max_size:]

        buffer = self._online_mppi_candidate_rank_buffer
        n = buffer["obs"].shape[0]
        batch_size = min(int(getattr(self.cfg, "online_mppi_candidate_rank_batch_size", 256)), n)
        margin = float(getattr(self.cfg, "mppi_candidate_selector_margin", None) or getattr(
            self.cfg, "online_mppi_candidate_rank_margin", 0.02
        ))
        good_weight = float(getattr(self.cfg, "online_mppi_candidate_good_weight", 1.0))
        bad_weight = float(getattr(self.cfg, "online_mppi_candidate_bad_weight", 1.0))

        last_stats = {}
        for _ in range(steps):
            idx = torch.randint(n, (batch_size,), device=self.device)
            raw_obs = {"policy": buffer["obs"][idx]}
            rank_obs = self._obs_normalizer(raw_obs) if self._obs_normalizer is not None else raw_obs
            with torch.no_grad():
                z0 = self._encoder(rank_obs).detach()
                policy = buffer["policy"][idx].detach()
                good = buffer["good"][idx].detach()
                bad = buffer["bad"][idx].detach()
                good = torch.where(buffer["good_mask"][idx].unsqueeze(-1), good, policy)

            usable_good = buffer["good_mask"][idx]

            if train_selector:
                policy_score = self._candidate_selector_score(z0, policy, policy)
                good_score = self._candidate_selector_score(z0, policy, good)
                bad_score = self._candidate_selector_score(z0, policy, bad)
            else:
                with torch.no_grad():
                    def predict_next(action: torch.Tensor) -> torch.Tensor:
                        return self._dynamics(z0=z0, controls=action.unsqueeze(1))[:, 1, :].detach()

                    policy_next = predict_next(policy)
                    good_next = predict_next(good)
                    bad_next = predict_next(bad)
                policy_score = self._reward(z0, policy_next, action=policy)
                good_score = self._reward(z0, good_next, action=good)
                bad_score = self._reward(z0, bad_next, action=bad)

            good_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            if usable_good.any():
                good_loss = F.relu(margin - (good_score[usable_good] - policy_score[usable_good])).mean()
            bad_loss = F.relu(margin - (policy_score - bad_score)).mean()
            rank_loss = good_weight * good_loss + bad_weight * bad_loss
            loss = coeff * rank_loss

            opt = self._candidate_selector_opt if train_selector else self._reward_opt
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                last_stats = {
                    "OnlineMPPICandidateRank/loss": loss.item(),
                    "OnlineMPPICandidateRank/rank_loss": rank_loss.item(),
                    "OnlineMPPICandidateRank/good_loss": good_loss.item(),
                    "OnlineMPPICandidateRank/bad_loss": bad_loss.item(),
                    "OnlineMPPICandidateRank/buffer_size": float(n),
                    "OnlineMPPICandidateRank/good_rate": buffer["good_mask"].float().mean().item(),
                    "OnlineMPPICandidateRank/train_selector": float(train_selector),
                    "OnlineMPPICandidateRank/policy_reward": policy_score.mean().item(),
                    "OnlineMPPICandidateRank/good_reward": good_score.mean().item(),
                    "OnlineMPPICandidateRank/bad_reward": bad_score.mean().item(),
                    "OnlineMPPICandidateRank/good_minus_policy": (good_score - policy_score).mean().item(),
                    "OnlineMPPICandidateRank/policy_minus_bad": (policy_score - bad_score).mean().item(),
                    "OnlineMPPICandidateRank/good_gt_policy": (good_score > policy_score).float().mean().item(),
                    "OnlineMPPICandidateRank/policy_gt_bad": (policy_score > bad_score).float().mean().item(),
                    "OnlineMPPICandidateRank/steps": float(steps),
                }
        return last_stats

    def _pretrain_dynamics_on_demo_actions(self, *, steps: int, batch_size: int):
        if self._bc_obs is None or self._bc_next_obs is None or self._bc_actions is None:
            return
        batch_size = min(int(batch_size), self._bc_obs.shape[0])
        was_training = self.planner.training
        self.planner.train()
        last = {}
        for _ in range(int(steps)):
            idx = torch.randint(self._bc_obs.shape[0], (batch_size,), device=self.device)
            loss, last = self._demo_action_dynamics_loss(idx)
            multi_loss, multi_last = self._demo_action_multistep_dynamics_loss(idx)
            loss = loss + multi_loss
            last.update(multi_last)
            self._dyn_opt.zero_grad(set_to_none=True)
            loss.backward()
            self._dyn_opt.step()
        if not was_training:
            self.planner.eval()
        print(
            "[MPAIL2Learner] Demo-action dynamics pretrained: "
            f"steps={int(steps)} batch_size={batch_size} "
            f"dyn_loss={last.get('dyn_loss', 0.0):.6f} "
            f"rank_loss={last.get('rank_loss', 0.0):.6f} "
            f"multi_loss={last.get('multistep_loss', 0.0):.6f} "
            f"multi_final={last.get('multistep_final_dist', 0.0):.6f} "
            f"margin_zero={last.get('margin_zero', 0.0):.6f} "
            f"margin_rand={last.get('margin_rand', 0.0):.6f}",
            flush=True,
        )

    def act(self, obs, critic_obs=None, vis_rollouts=False):
        '''Performs forward pass on MPPI and collects discriminator logits for MPAIL
        TODO: Critic obs?'''

        # Do NOT normalize obs for planner - planner uses raw observations
        # Normalization is only applied when feeding to value function

        # Compute the actions and values
        if critic_obs is None:
            critic_obs = obs

        with torch.no_grad():
            self._last_mppi_action_stats = {}
            self._last_mppi_action_tensors = {}
            if self._act_calls < self.cfg.demo_action_bc_direct_steps:
                actions = self._direct_demo_action(obs)
            else:
                base_plan = None
                policy_actions = None
                if getattr(self.cfg, "mppi_residual_base_policy", False):
                    base_plan = self._bc_policy_plan(obs)
                    policy_actions = base_plan[:, 0]
                action_library_plans = self._mppi_action_library_plans(obs, base_plan=base_plan)
                mppi_actions = self.planner.act(
                    obs,
                    vis_rollouts=vis_rollouts,
                    base_plan=base_plan,
                    action_library_plans=action_library_plans,
                )
                self._append_mppi_elite_adversarial_transitions()
                best_candidate_stats = {}
                best_candidate_accept = None
                if getattr(self.cfg, "mppi_best_candidate_action", False):
                    if policy_actions is None:
                        policy_actions = self._direct_demo_action(obs)
                    mppi_actions, best_candidate_stats, best_candidate_accept = (
                        self._mppi_best_candidate_action(policy_actions, obs=obs, action_library_plans=action_library_plans)
                    )
                blend = float(getattr(self.cfg, "mppi_policy_blend", 1.0))
                if blend < 1.0:
                    if policy_actions is None:
                        policy_actions = self._direct_demo_action(obs)
                    delta = mppi_actions - policy_actions
                    raw_delta = delta
                    raw_delta_norm = raw_delta.norm(dim=-1)
                    raw_delta_max = raw_delta.abs().amax(dim=-1)
                    cos = F.cosine_similarity(mppi_actions, policy_actions, dim=-1)
                    l2_scale = torch.ones_like(raw_delta_norm)
                    delta_l2_max = getattr(self.cfg, "mppi_policy_delta_l2_max", None)
                    if delta_l2_max is not None:
                        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        delta_scale = (float(delta_l2_max) / delta_norm).clamp(max=1.0)
                        delta = delta * delta_scale
                        l2_scale = delta_scale.squeeze(-1)
                    actions = (policy_actions + blend * delta).clamp(-1.0, 1.0)
                    delta_max = getattr(self.cfg, "mppi_policy_delta_max", None)
                    unsafe = torch.zeros((policy_actions.shape[0], 1), device=policy_actions.device, dtype=torch.bool)
                    if delta_max is not None:
                        unsafe = delta.abs().amax(dim=-1, keepdim=True) > float(delta_max)
                        actions = torch.where(unsafe, policy_actions, actions)
                    reward_gate_accept = None
                    policy_next_for_gate = None
                    action_next_for_gate = None
                    z0_for_gate = None
                    reward_gate_margin = getattr(self.cfg, "mppi_reward_gate_margin", None)
                    if reward_gate_margin is not None:
                        rank_obs = obs
                        if getattr(self.planner, "_obs_normalizer", None) is not None:
                            rank_obs = self.planner._obs_normalizer(rank_obs)
                        z0_for_gate = self._encoder(rank_obs)
                        policy_next_for_gate = self._dynamics(
                            z0=z0_for_gate, controls=policy_actions.unsqueeze(1)
                        )[:, 1, :]
                        action_next_for_gate = self._dynamics(z0=z0_for_gate, controls=actions.unsqueeze(1))[:, 1, :]
                        policy_reward = self._reward(z0_for_gate, policy_next_for_gate, action=policy_actions)
                        action_reward = self._reward(z0_for_gate, action_next_for_gate, action=actions)
                        reward_gate_accept = action_reward >= policy_reward + float(reward_gate_margin)
                        unsafe = unsafe | (~reward_gate_accept).unsqueeze(-1)
                        actions = torch.where(unsafe, policy_actions, actions)
                    progress_gate_accept = None
                    progress_margin = getattr(self.cfg, "mppi_progress_gate_margin", None)
                    if progress_margin is not None:
                        if policy_next_for_gate is None or action_next_for_gate is None:
                            rank_obs = obs
                            if getattr(self.planner, "_obs_normalizer", None) is not None:
                                rank_obs = self.planner._obs_normalizer(rank_obs)
                            z0_for_gate = self._encoder(rank_obs)
                            policy_next_for_gate = self._dynamics(
                                z0=z0_for_gate, controls=policy_actions.unsqueeze(1)
                            )[:, 1, :]
                            action_next_for_gate = self._dynamics(
                                z0=z0_for_gate, controls=actions.unsqueeze(1)
                            )[:, 1, :]
                        if getattr(self.planner, "_obs_normalizer", None) is not None:
                            policy_next_raw = self.planner._obs_normalizer.inverse({"policy": policy_next_for_gate})[
                                "policy"
                            ]
                            action_next_raw = self.planner._obs_normalizer.inverse({"policy": action_next_for_gate})[
                                "policy"
                            ]
                        else:
                            policy_next_raw = policy_next_for_gate
                            action_next_raw = action_next_for_gate
                        policy_dist = policy_next_raw[..., 0:3].norm(dim=-1)
                        action_dist = action_next_raw[..., 0:3].norm(dim=-1)
                        progress_gate_accept = action_dist <= policy_dist + float(progress_margin)
                        unsafe = unsafe | (~progress_gate_accept).unsqueeze(-1)
                        actions = torch.where(unsafe, policy_actions, actions)
                    cos_gate_accept = None
                    cos_min = getattr(self.cfg, "mppi_policy_cos_min", None)
                    if cos_min is not None:
                        cos_gate_accept = cos >= float(cos_min)
                        unsafe = unsafe | (~cos_gate_accept).unsqueeze(-1)
                        actions = torch.where(unsafe, policy_actions, actions)
                    applied_delta = actions - policy_actions
                    gate_accept = ~unsafe.squeeze(-1)
                    self._last_mppi_action_stats = {
                        "MPPI/action_library_size": float(
                            0 if action_library_plans is None else action_library_plans.shape[1]
                        ),
                        "MPPI/policy_delta_l2": raw_delta_norm.mean().item(),
                        "MPPI/policy_delta_l2_max": raw_delta_norm.max().item(),
                        "MPPI/policy_delta_absmax": raw_delta_max.mean().item(),
                        "MPPI/policy_delta_absmax_max": raw_delta_max.max().item(),
                        "MPPI/policy_cos": cos.mean().item(),
                        "MPPI/policy_cos_min": cos.min().item(),
                        "MPPI/blend": blend,
                        "MPPI/gate_fallback_rate": unsafe.float().mean().item(),
                        "MPPI/gate_accept_rate": gate_accept.float().mean().item(),
                        "MPPI/l2_scale_mean": l2_scale.mean().item(),
                        "MPPI/l2_scale_min": l2_scale.min().item(),
                        "MPPI/applied_delta_l2": applied_delta.norm(dim=-1).mean().item(),
                        "MPPI/applied_delta_absmax": applied_delta.abs().amax(dim=-1).mean().item(),
                    }
                    self._last_mppi_action_stats.update(best_candidate_stats)
                    if best_candidate_accept is not None:
                        self._last_mppi_action_stats["MPPI/best_candidate_gate_accept_rate"] = (
                            best_candidate_accept.float().mean().item()
                        )
                    if reward_gate_accept is not None:
                        self._last_mppi_action_stats["MPPI/reward_gate_accept_rate"] = (
                            reward_gate_accept.float().mean().item()
                        )
                    if progress_gate_accept is not None:
                        self._last_mppi_action_stats["MPPI/progress_gate_accept_rate"] = (
                            progress_gate_accept.float().mean().item()
                        )
                    if cos_gate_accept is not None:
                        self._last_mppi_action_stats["MPPI/cos_gate_accept_rate"] = (
                            cos_gate_accept.float().mean().item()
                        )
                    for dim in range(applied_delta.shape[-1]):
                        self._last_mppi_action_stats[f"MPPI/applied_delta_dim{dim}_mean"] = (
                            applied_delta[:, dim].mean().item()
                        )
                        self._last_mppi_action_stats[f"MPPI/applied_delta_dim{dim}_abs_mean"] = (
                            applied_delta[:, dim].abs().mean().item()
                        )
                    self._last_mppi_action_tensors = {
                        "mppi": mppi_actions.detach(),
                        "policy": policy_actions.detach(),
                        "blended": (policy_actions + blend * delta).clamp(-1.0, 1.0).detach(),
                        "executed": actions.detach(),
                        "gate_accept": gate_accept.detach(),
                    }
                else:
                    actions = mppi_actions
                    self._last_mppi_action_tensors = {
                        "mppi": mppi_actions.detach(),
                        "executed": actions.detach(),
                    }
            self._act_calls += 1

        self.transition.actions = actions
        # need to record obs and critic_obs before env.step()
        # Store observations (both dict and tensor cases)
        self.transition.observations = obs
        self.transition.critic_observations = obs

        return actions

    def process_env_step(self, rewards, dones, infos, next_obs, valid_transitions=None) -> Dict[str, float]:

        with torch.no_grad():

            # Record the rewards and dones
            # Note: we clone here because later on we bootstrap the rewards based on timeouts
            self.transition.dones = dones
            if valid_transitions is None:
                valid_transitions = torch.ones_like(dones, dtype=torch.bool, device=self.device)
            self.transition.valid = valid_transitions

            #################################################################################################
            self.transition.rewards = rewards # NOTE: THESE ARE NOT TO BE USED IN LEARNING - JUST FOR LOGGING
            #################################################################################################

            # Compute policy stats
            stats = self.planner.compute_stats()

            # Record the transition
            self.storage.add_transitions(self.transition)
            self.transition.clear()
            # Convert dones to indices for reset
            reset_inds = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if reset_inds.numel() > 0:
                self.planner.reset(reset_inds)
            # If no environments are done, don't reset anything

        return stats

    @property
    def _total_batch_size(self):
        return self.storage.num_steps_per_env * self.storage.num_envs

    def _create_target_value_network(self):
        """Create target value network as a deep copy of the main value network"""
        target_network = copy.deepcopy(self._value)
        # Disable gradients for target network
        for param in target_network.parameters():
            param.requires_grad_(False)

        target_network.eval()  # Set to eval mode

        for target_param, main_param in zip(target_network.parameters(), self._value.parameters()):
            target_param.data.copy_(main_param.data)

        return target_network

    def _soft_update_target_value(self):
        """Soft update using Polyak averaging: target = (1-tau) * target + tau * main"""
        for target_param, main_param in zip(self._value_target.parameters(), self._value.parameters()):
            target_param.data.copy_(
                (1.0 - self._polyak_tau) * target_param.data + self._polyak_tau * main_param.data
            )

    def update(self, iteration: int) -> Dict[str, float]:

        self.storage.process_replay_buffer() # Move episode storage to replay buffer
        if self.cfg.skip_updates:
            return {"Learner/skip_updates": 1.0}

        # Demonstrations are dictionaries where each group has [N, 2, *obs_shape]
        # Sample consecutive subtrajectories of length loss_horizon
        demo_dataset = DictDemoDataset(
            self._demonstrations,
            horizon=self.cfg.loss_horizon,
            device=self.device,
            obs_normalizer=self._obs_normalizer
        )

        _num_flat_trajs = self.storage.replay_batch_size // self.cfg.loss_horizon
        demo_batch_size = _num_flat_trajs  # Number of subtrajectories to sample

        # Calculate number of updates based on replay ratio and new data collected
        num_updates = max(1, int(self.replay_ratio * self._total_batch_size))

        # Use custom batch generator to avoid duplicate sampling
        demo_generator = demo_dataset.mini_traj_batch_generator(
            batch_size=demo_batch_size,
            num_epochs=num_updates
        )

        traj_generator = self.storage.mini_traj_batch_generator(
            self.cfg.loss_horizon,
            num_epochs=num_updates,
            return_demo_mask=True,
        )

        self._mean_stats  = defaultdict(float)
        self._tot_stats = defaultdict(float)

        actual_updates = 0

        for (
            (replay_obs_traj, replay_actions_traj, replay_next_obs_traj, dones_traj, replay_is_demo_traj),
            (demo_obs_traj, demo_next_obs_traj)
        ) in zip(traj_generator, demo_generator):

            # Flatten dict observations
            demo_obs_flat = {k: v.flatten(start_dim=0, end_dim=1) for k, v in demo_obs_traj.items()}
            demo_next_obs_flat = {k: v.flatten(start_dim=0, end_dim=1) for k, v in demo_next_obs_traj.items()}

            # (1) UPDATE DYNAMICS (agent data + demo data combined)
            _start = time.time()
            self._mean_stats['Dyn/mean_loss'] += self.update_dynamics(
                obs_batch_traj=replay_obs_traj,
                next_obs_batch_traj=replay_next_obs_traj,
                actions_batch_traj=replay_actions_traj,
                demo_obs_traj=demo_obs_traj,
                demo_next_obs_traj=demo_next_obs_traj,
            )
            self._tot_stats['Dyn/learn_time'] += time.time() - _start

            # Encode observations to latent space
            with torch.no_grad():
                replay_latent_traj = self._encoder(replay_obs_traj)
                replay_next_latent_traj = self._encoder(replay_next_obs_traj)
                demo_latent_flat = self._encoder(demo_obs_flat)
                demo_next_latent_flat = self._encoder(demo_next_obs_flat)

            # Flatten trajectories for non-trajectory-based updates
            _num_flat_trajs = self.storage.replay_batch_size // self.cfg.loss_horizon
            latent_batch_flat = replay_latent_traj[:_num_flat_trajs].flatten(start_dim=0, end_dim=1)
            next_latent_batch_flat = replay_next_latent_traj[:_num_flat_trajs].flatten(start_dim=0, end_dim=1)
            actions_batch_flat = replay_actions_traj[:_num_flat_trajs].flatten(start_dim=0, end_dim=1)
            replay_is_demo_flat = None
            if replay_is_demo_traj is not None:
                replay_is_demo_flat = replay_is_demo_traj[:_num_flat_trajs].flatten(start_dim=0, end_dim=1).squeeze(-1).bool()
                self._mean_stats['Replay/demo_success_fraction'] += replay_is_demo_flat.float().mean().item()

            # Create flattened terminal mask for value learning
            # terminal_mask: [batch_size] - True if last timestep is terminal
            # We need: [batch_size * horizon] where only last timestep of each traj can be terminal
            terminal_mask_flat = None
            if self.cfg.use_terminations and dones_traj is not None:
                horizon = replay_latent_traj.shape[1]
                batch_size = min(_num_flat_trajs, replay_latent_traj.shape[0])
                terminal_mask_flat = torch.zeros(batch_size, horizon, device=self.device, dtype=torch.bool)
                terminal_mask_flat[:, -1] = dones_traj[:batch_size]  # Only last timestep can be terminal
                terminal_mask_flat = terminal_mask_flat.flatten()  # [batch_size * horizon]
                # Log how many terminals we see
                self._mean_stats['Value/num_terminals'] += terminal_mask_flat.sum().item()

            # (2) UPDATE REWARD (skipped if reward_update_every > 1 and not on schedule)
            _start = time.time()
            if getattr(self.cfg, "skip_reward_updates", False):
                self._mean_stats['Reward/skipped'] += 1.0
            elif actual_updates % self.cfg.reward_update_every == 0:
                _reward_stats = self.update_reward(
                    demo_latent_batch=demo_latent_flat,
                    demo_next_latent_batch=demo_next_latent_flat,
                    latent_batch=latent_batch_flat,
                    next_latent_batch=next_latent_batch_flat,
                    action_batch=actions_batch_flat,
                    gen_negative_mask=None if replay_is_demo_flat is None else ~replay_is_demo_flat,
                )
                for key, value in _reward_stats.items():
                    self._mean_stats[key] += value
                if self.cfg.reward_center_for_planning:
                    center = self._reward_stats_to_center(_reward_stats)
                    self._reward_center.copy_(torch.as_tensor(center, device=self.device, dtype=self.dtype))
                    self.planner.set_reward_center(self._reward_center)
                    self._mean_stats['Reward/center'] += self._reward_center.item()
            self._tot_stats['Reward/learn_time'] += time.time() - _start

            # (3) UPDATE ONLINE IQ-LEARN / IQ REWARD DISTILLATION
            _start = time.time()
            iq_online_steps = int(getattr(self.cfg, "iqlearn_online_steps", 0))
            for _ in range(iq_online_steps):
                _iq_stats = self.update_iqlearn_online(
                    latent_batch=latent_batch_flat,
                    action_batch=actions_batch_flat,
                    next_latent_batch=next_latent_batch_flat,
                    replay_is_demo=replay_is_demo_flat,
                    terminal_mask=terminal_mask_flat,
                    batch_size=int(getattr(self.cfg, "iqlearn_online_batch_size", 512)),
                )
                for key, value in _iq_stats.items():
                    self._mean_stats[key] += value / max(1, iq_online_steps)

            iq_distill_steps = int(getattr(self.cfg, "iqlearn_online_reward_distill_steps", 0))
            for _ in range(iq_distill_steps):
                _iq_reward_stats = self.update_iqlearn_reward_distill_online(
                    latent_batch=latent_batch_flat,
                    action_batch=actions_batch_flat,
                    next_latent_batch=next_latent_batch_flat,
                    replay_is_demo=replay_is_demo_flat,
                    terminal_mask=terminal_mask_flat,
                    batch_size=int(getattr(self.cfg, "iqlearn_online_reward_distill_batch_size", 512)),
                )
                for key, value in _iq_reward_stats.items():
                    self._mean_stats[key] += value / max(1, iq_distill_steps)
            self._tot_stats['IQOnline/learn_time'] += time.time() - _start

            # (4) UPDATE VALUE
            _start = time.time()
            self._mean_stats['Value/mean_loss'] += self.update_value(
                latent_batch=latent_batch_flat,
                action_batch=actions_batch_flat,
                next_latent_batch=next_latent_batch_flat,
                learner_cfg=self.value_learner_cfg,
                terminal_mask=terminal_mask_flat,
            )
            self._tot_stats['Value/learn_time'] += time.time() - _start

            # (5) UPDATE POLICY
            _start = time.time()
            if getattr(self.cfg, "skip_policy_updates", False) or iteration < getattr(self.cfg, "policy_update_start_iter", 0):
                self._mean_stats['Policy/skipped'] += 1.0
            else:
                self._mean_stats['Policy/mean_loss'] += self.update_policy( # TODO: dont duplicate encoding
                    latent_batch=latent_batch_flat,
                )
            self._tot_stats['Policy/learn_time'] += time.time() - _start

            actual_updates += 1
            self._mean_stats['Data/mean_latent_norm'] += latent_batch_flat.norm(dim=-1).mean().item()

            if actual_updates == 1:
                reward_diag = self._compute_reward_replay_diagnostics(demo_latent_flat, demo_next_latent_flat)
                for key, value in reward_diag.items():
                    self._tot_stats[key] = value
                value_action_diag = self._compute_value_action_diagnostics(latent_batch_flat, actions_batch_flat)
                for key, value in value_action_diag.items():
                    self._tot_stats[key] = value

        # Average stats over updates
        for key in self._mean_stats.keys():
            self._mean_stats[key] /= max(1, actual_updates)

        # Store training info
        self._tot_stats['Training/num_updates'] = actual_updates
        self._tot_stats['Training/replay_ratio'] = self.replay_ratio

        # Log how much data is in the replay buffer
        self._tot_stats['Storage/replay_size'] = self.storage.num_trajectories_stored * self.storage.num_steps_per_env
        self._tot_stats.update(self.storage.compute_replay_stats())

        # -- Clear the storage
        self.storage.clear()

        self._mean_stats.update(self._tot_stats)

        return self._mean_stats

    def _compute_reward_replay_diagnostics(self, demo_latent_flat, demo_next_latent_flat) -> Dict[str, float]:
        """Score oldest/newest replay slices under the current reward for distribution drift diagnostics."""
        if not self.storage.has_replay_data or self.storage.num_trajectories_stored == 0:
            return {}

        with torch.no_grad():
            horizon = min(self.cfg.loss_horizon, self.storage.num_steps_per_env - 1)
            if horizon <= 0:
                return {}

            num_slices = min(2, self.storage.num_trajectories_stored)
            slices = {
                "oldest": slice(0, num_slices),
                "newest": slice(self.storage.num_trajectories_stored - num_slices, self.storage.num_trajectories_stored),
            }

            stats = {}
            for name, traj_slice in slices.items():
                obs_seq = {k: v[:horizon, traj_slice].transpose(0, 1)
                           for k, v in self.storage.replay_data["obs"].items()}
                next_obs_seq = {k: v[1:horizon + 1, traj_slice].transpose(0, 1)
                                for k, v in self.storage.replay_data["obs"].items()}
                if self._obs_normalizer is not None:
                    obs_seq = self._obs_normalizer(obs_seq)
                    next_obs_seq = self._obs_normalizer(next_obs_seq)
                z = self._encoder(obs_seq).flatten(start_dim=0, end_dim=1)
                next_z = self._encoder(next_obs_seq).flatten(start_dim=0, end_dim=1)
                stats[f"Reward/diag_{name}_replay_score"] = self._reward(z, next_z).mean().item()

            stats["Reward/diag_demo_score"] = self._reward(demo_latent_flat, demo_next_latent_flat).mean().item()
            return stats

    def _compute_value_action_diagnostics(self, latent_batch_flat, actions_batch_flat) -> Dict[str, float]:
        """Compare Q and dynamics sensitivity for executed, zero, and random actions."""
        with torch.no_grad():
            sample_count = min(512, latent_batch_flat.shape[0])
            z = latent_batch_flat[:sample_count]
            executed = actions_batch_flat[:sample_count]
            zeros = torch.zeros_like(executed)
            random_actions = torch.empty_like(executed).uniform_(-1.0, 1.0)

            q_exec_all = self._value(z, executed, return_type="all")
            q_zero_all = self._value(z, zeros, return_type="all")
            q_rand_all = self._value(z, random_actions, return_type="all")

            next_exec = self._dynamics(z0=z, controls=executed.unsqueeze(1))[:, 1, :]
            next_zero = self._dynamics(z0=z, controls=zeros.unsqueeze(1))[:, 1, :]
            next_rand = self._dynamics(z0=z, controls=random_actions.unsqueeze(1))[:, 1, :]

            return {
                "ValueDiag/q_exec_mean": q_exec_all.mean().item(),
                "ValueDiag/q_zero_mean": q_zero_all.mean().item(),
                "ValueDiag/q_rand_mean": q_rand_all.mean().item(),
                "ValueDiag/q_exec_minus_zero": (q_exec_all.mean(0) - q_zero_all.mean(0)).mean().item(),
                "ValueDiag/q_rand_minus_zero": (q_rand_all.mean(0) - q_zero_all.mean(0)).mean().item(),
                "ValueDiag/q_ensemble_std_exec": q_exec_all.std(0).mean().item(),
                "ValueDiag/q_ensemble_std_zero": q_zero_all.std(0).mean().item(),
                "Dyn/action_sensitivity_exec_vs_zero": (next_exec - next_zero).norm(dim=-1).mean().item(),
                "Dyn/action_sensitivity_rand_vs_zero": (next_rand - next_zero).norm(dim=-1).mean().item(),
                "ActionDiag/replay_exec_norm": executed.norm(dim=-1).mean().item(),
                "ActionDiag/random_norm": random_actions.norm(dim=-1).mean().item(),
            }

    def update_dynamics(
        self,
        obs_batch_traj: torch.Tensor,
        next_obs_batch_traj: torch.Tensor,
        actions_batch_traj: torch.Tensor,
        demo_obs_traj: Optional[Dict[str, torch.Tensor]] = None,
        demo_next_obs_traj: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Concatenate agent + demo trajectories so encoder sees expert data too
        if demo_obs_traj is not None and demo_next_obs_traj is not None:
            shared_keys = [k for k in obs_batch_traj if k in demo_obs_traj]
            obs_combined = {k: torch.cat([obs_batch_traj[k], demo_obs_traj[k]], dim=0)
                            for k in shared_keys}
            next_obs_combined = {k: torch.cat([next_obs_batch_traj[k], demo_next_obs_traj[k]], dim=0)
                                 for k in shared_keys}
            # Demo has no actions; use zeros as placeholder (only JEP loss on agent portion)
            demo_actions = torch.zeros(
                demo_obs_traj[next(iter(demo_obs_traj))].shape[0],
                actions_batch_traj.shape[1],
                actions_batch_traj.shape[2],
                device=self.device, dtype=self.dtype,
            )
            actions_combined = torch.cat([actions_batch_traj, demo_actions], dim=0)
            agent_batch_size = obs_batch_traj[next(iter(obs_batch_traj))].shape[0]
        else:
            obs_combined = obs_batch_traj
            next_obs_combined = next_obs_batch_traj
            actions_combined = actions_batch_traj
            agent_batch_size = obs_batch_traj[next(iter(obs_batch_traj))].shape[0]

        # Encode the combined batch (agent + demo); target next-latent is detached
        latent_combined_traj = self._encoder(obs_combined)
        with torch.no_grad():
            next_latent_combined_traj = self._encoder(next_obs_combined)

        _H = actions_combined.shape[1]
        rho = self.dynamics_learner_cfg.rho
        _rhos = rho ** torch.arange(_H, device=self.device)

        z0 = latent_combined_traj[:, 0, :]
        pred_latents = self._dynamics(z0=z0, controls=actions_combined)

        # JEP loss on agent portion only (demo has placeholder zero actions)
        _jep_se = (pred_latents[:agent_batch_size, 1:, :] -
                   next_latent_combined_traj[:agent_batch_size].detach()).pow(2)
        agent_loss = (_rhos[..., None] * _jep_se).mean()
        loss = self.cfg.dyn_agent_coeff * agent_loss
        self._mean_stats['Dyn/agent_loss'] += agent_loss.item()

        aux_total, aux_stats = self._task_dynamics_aux_loss(
            obs_combined["policy"][:agent_batch_size],
            pred_latents[:agent_batch_size, 1:, :],
            next_latent_combined_traj[:agent_batch_size].detach(),
        )
        loss = loss + aux_total
        self._mean_stats['Dyn/task_obs_loss'] += aux_stats["task_obs_loss"]
        self._mean_stats['Dyn/progress_loss'] += aux_stats["progress_loss"]
        self._mean_stats['Dyn/progress_sign_loss'] += aux_stats["progress_sign_loss"]
        self._mean_stats['Dyn/progress_over_loss'] += aux_stats["progress_over_loss"]
        self._mean_stats['Dyn/progress_sign_acc_train'] += aux_stats["progress_sign_acc_train"]

        counter_coeff = float(getattr(self.cfg, "dyn_counterfactual_progress_coeff", 0.0))
        if counter_coeff > 0.0:
            z_agent = latent_combined_traj[:agent_batch_size, 0, :]
            cur_agent = obs_combined["policy"][:agent_batch_size]
            verified_agent = pred_latents[:agent_batch_size, 1:, :].detach()
            zero_actions = torch.zeros_like(actions_batch_traj[:agent_batch_size])
            rand_actions = torch.empty_like(actions_batch_traj[:agent_batch_size]).uniform_(-1.0, 1.0)
            zero_next = self._dynamics(z0=z_agent, controls=zero_actions)[:, 1:, :]
            rand_next = self._dynamics(z0=z_agent, controls=rand_actions)[:, 1:, :]
            zero_cf_loss, zero_cf_stat = self._counterfactual_progress_conservative_loss(
                cur_agent, verified_agent, zero_next
            )
            rand_cf_loss, rand_cf_stat = self._counterfactual_progress_conservative_loss(
                cur_agent, verified_agent, rand_next
            )
            loss = loss + zero_cf_loss + rand_cf_loss
            self._mean_stats['Dyn/counterfactual_zero_progress_loss'] += zero_cf_stat
            self._mean_stats['Dyn/counterfactual_rand_progress_loss'] += rand_cf_stat

        # Decoder for visualization if specified
        if self._decoder:
            _recon_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for obs_key in next_obs_batch_traj:
                _recon_loss += F.mse_loss(
                    self._decoder(next_latent_combined_traj[:agent_batch_size].detach())[obs_key],
                    next_obs_batch_traj[obs_key],
                )
        else:
            _recon_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        self._mean_stats['Dyn/recon_loss'] += _recon_loss.item()

        demo_action_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        demo_action_rank_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if (
            self._bc_obs is not None
            and self._bc_next_obs is not None
            and self._bc_actions is not None
            and (
                self.cfg.demo_action_dyn_coeff > 0
                or self.cfg.demo_action_rank_coeff > 0
                or getattr(self.cfg, "demo_action_multistep_dyn_coeff", 0.0) > 0
            )
        ):
            bc_batch_size = min(agent_batch_size * _H, self._bc_obs.shape[0])
            bc_idx = torch.randint(self._bc_obs.shape[0], (bc_batch_size,), device=self.device)
            demo_action_total, demo_action_stats = self._demo_action_dynamics_loss(bc_idx)
            multi_total, multi_stats = self._demo_action_multistep_dynamics_loss(bc_idx)
            demo_action_total = demo_action_total + multi_total
            loss = loss + demo_action_total
            self._mean_stats['Dyn/demo_action_loss'] += demo_action_stats["dyn_loss"]
            self._mean_stats['Dyn/demo_action_rank_loss'] += demo_action_stats["rank_loss"]
            self._mean_stats['Dyn/demo_action_task_obs_loss'] += demo_action_stats["task_obs_loss"]
            self._mean_stats['Dyn/demo_action_progress_loss'] += demo_action_stats["progress_loss"]
            self._mean_stats['Dyn/demo_action_progress_sign_loss'] += demo_action_stats["progress_sign_loss"]
            self._mean_stats['Dyn/demo_action_progress_over_loss'] += demo_action_stats["progress_over_loss"]
            self._mean_stats['Dyn/demo_action_progress_sign_acc_train'] += demo_action_stats["progress_sign_acc_train"]
            self._mean_stats['Dyn/demo_action_teacher_dist'] += demo_action_stats["teacher_dist"]
            self._mean_stats['Dyn/demo_action_zero_dist'] += demo_action_stats["zero_dist"]
            self._mean_stats['Dyn/demo_action_rand_dist'] += demo_action_stats["rand_dist"]
            self._mean_stats['Dyn/demo_action_margin_zero'] += demo_action_stats["margin_zero"]
            self._mean_stats['Dyn/demo_action_margin_rand'] += demo_action_stats["margin_rand"]
            self._mean_stats['Dyn/demo_action_multistep_loss'] += multi_stats["multistep_loss"]
            self._mean_stats['Dyn/demo_action_multistep_final_dist'] += multi_stats["multistep_final_dist"]

        self._dyn_opt.zero_grad()
        loss.backward()
        if self._decoder:
            _recon_loss.backward()
        self._dyn_opt.step()

        return loss.item()

    def update_reward(
        self,
        latent_batch: torch.Tensor,
        next_latent_batch: torch.Tensor,
        demo_latent_batch: torch.Tensor,
        demo_next_latent_batch: torch.Tensor,
        action_batch: Optional[torch.Tensor] = None,
        gen_negative_mask: Optional[torch.Tensor] = None,
    ):
        '''Trains reward network using WGAN loss (discriminator-style adversarial training)'''

        replay_demo_fraction = 0.0
        if gen_negative_mask is not None:
            gen_negative_mask = gen_negative_mask.to(device=latent_batch.device, dtype=torch.bool)
            replay_demo_fraction = 1.0 - gen_negative_mask.float().mean().item()
            if gen_negative_mask.any():
                latent_batch = latent_batch[gen_negative_mask]
                next_latent_batch = next_latent_batch[gen_negative_mask]
                if action_batch is not None:
                    action_batch = action_batch[gen_negative_mask]
            else:
                return {
                    "Reward/skipped_all_demo_replay": 1.0,
                    "Reward/replay_demo_success_fraction": replay_demo_fraction,
                }

        if demo_latent_batch.shape[0] != latent_batch.shape[0]:
            demo_idx = torch.randint(demo_latent_batch.shape[0], (latent_batch.shape[0],), device=self.device)
            demo_latent_batch = demo_latent_batch[demo_idx]
            demo_next_latent_batch = demo_next_latent_batch[demo_idx]

        demo_action_batch = None
        if getattr(self._reward.cfg, "use_action", False) and self._bc_actions is not None:
            demo_batch_size = latent_batch.shape[0]
            bc_idx = torch.randint(self._bc_obs.shape[0], (demo_batch_size,), device=self.device)
            bc_obs = {"policy": self._bc_obs[bc_idx]}
            bc_next_obs = {"policy": self._bc_next_obs[bc_idx]}
            if self._obs_normalizer is not None:
                bc_obs = self._obs_normalizer(bc_obs)
                bc_next_obs = self._obs_normalizer(bc_next_obs)
            demo_latent_batch = self._encoder(bc_obs)
            with torch.no_grad():
                demo_next_latent_batch = self._encoder(bc_next_obs)
            demo_action_batch = self._bc_actions[bc_idx]

        _demo_rewards = self._reward(demo_latent_batch, demo_next_latent_batch, action=demo_action_batch)
        _gen_rewards = self._reward(latent_batch, next_latent_batch, action=action_batch)

        # WGAN loss: maximize demo rewards, minimize generator rewards
        loss = -_demo_rewards.mean() + _gen_rewards.mean()
        elite_adv_rewards = None
        elite_adv_coeff = float(getattr(self.cfg, "mppi_elite_adversarial_reward_coeff", 0.0))
        if elite_adv_coeff > 0.0 and self._mppi_elite_adversarial_buffer is not None:
            buffer = self._mppi_elite_adversarial_buffer
            n = buffer["state"].shape[0]
            if n > 0:
                batch_size = min(int(getattr(self.cfg, "mppi_elite_adversarial_batch_size", 256)), n)
                idx = torch.randint(n, (batch_size,), device=self.device)
                elite_state = buffer["state"][idx].detach()
                elite_next_state = buffer["next_state"][idx].detach()
                elite_action = buffer["action"][idx].detach()
                elite_adv_rewards = self._reward(elite_state, elite_next_state, action=elite_action)
                loss = loss + elite_adv_coeff * elite_adv_rewards.mean()

        train_stats = {}
        # add gradient penalty if coefficient is specified
        gp_loss = self.reward_learner_cfg.gp_coeff * self._compute_gradient_penalty(
            expert_state=demo_latent_batch.detach(),
            expert_next_state=demo_next_latent_batch.detach(),
            expert_action=None if demo_action_batch is None else demo_action_batch.detach(),
            gen_state=latent_batch.detach(),
            gen_next_state=next_latent_batch.detach(),
            gen_action=None if action_batch is None else action_batch.detach(),
            stats_dict=train_stats,
        )
        loss += gp_loss

        reward_rank_stats = None
        if (
            getattr(self.cfg, "demo_action_reward_rank_coeff", 0.0) > 0
            and self._bc_obs is not None
            and self._bc_actions is not None
        ):
            rank_batch_size = min(latent_batch.shape[0], self._bc_obs.shape[0])
            rank_idx = torch.randint(self._bc_obs.shape[0], (rank_batch_size,), device=self.device)
            reward_rank_loss, reward_rank_stats = self._demo_action_reward_rank_loss(rank_idx)
            loss += reward_rank_loss

        # R1 regularization: penalize large absolute reward scores to prevent saturation
        r1_loss = self.reward_learner_cfg.r1_coeff * (
            _demo_rewards.pow(2).mean() + _gen_rewards.pow(2).mean()
        )
        loss += r1_loss

        with torch.no_grad():
            # FOR STATS LOGGING
            _output = torch.cat((
                _demo_rewards,
                _gen_rewards
            ))
            _target = torch.cat(( # [ 1 \\ 0 ]
                torch.ones_like(_demo_rewards),
                torch.zeros_like(_gen_rewards)
            ))

        # do gradient step
        self._reward_opt.zero_grad()
        loss.backward()
        self._reward_opt.step()

        # Compute training stats
        with torch.no_grad():

            train_stats.update(compute_disc_train_stats(
                _output, _target, loss, from_probs=False
            ))

            train_stats.update({
                "mean_demo_reward": _demo_rewards.mean().item(),
                "mean_gen_reward": _gen_rewards.mean().item(),
                "demo_std": _demo_rewards.std().item(),
                "replay_demo_success_fraction": replay_demo_fraction,
            })
            if elite_adv_rewards is not None:
                train_stats.update({
                    "mean_mppi_elite_adv_reward": elite_adv_rewards.mean().item(),
                    "mppi_elite_adv_coeff": elite_adv_coeff,
                    "mppi_elite_adv_buffer_size": float(self._mppi_elite_adversarial_buffer["state"].shape[0]),
                })

            if self.reward_learner_cfg.gp_coeff is not None and self.reward_learner_cfg.gp_coeff > 0:
                train_stats["gp_loss"] = gp_loss.item()

            if self.reward_learner_cfg.r1_coeff is not None and self.reward_learner_cfg.r1_coeff > 0:
                train_stats["r1_loss"] = r1_loss.item()
            if reward_rank_stats is not None:
                for key, value in reward_rank_stats.items():
                    train_stats[f"demo_action_reward_{key}"] = value

        _train_stats = {}
        for key, value in train_stats.items():
            _train_stats[f"Reward/{key}"] = value

        _train_stats['Reward/learner_latent_std'] = latent_batch.std(-2).mean().item()
        _train_stats['Reward/demo_latent_std'] = demo_latent_batch.std(-2).mean().item()

        return _train_stats

    def _reward_stats_to_center(self, reward_stats: Dict[str, float]) -> float:
        demo_mean = reward_stats["Reward/mean_demo_reward"]
        gen_mean = reward_stats["Reward/mean_gen_reward"]
        if self.cfg.reward_center_mode == "demo_mean":
            return demo_mean
        if self.cfg.reward_center_mode != "midpoint":
            raise ValueError(f"Unknown reward_center_mode: {self.cfg.reward_center_mode}")
        return 0.5 * (demo_mean + gen_mean)

    def _calibrate_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.cfg.reward_center_for_planning:
            rewards = rewards - self._reward_center.to(device=rewards.device, dtype=rewards.dtype)
        clip = self.cfg.reward_clip_for_planning
        if clip is not None and clip > 0:
            rewards = rewards.clamp(-clip, clip)
        return rewards

    def _compute_gradient_penalty(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        expert_action: Optional[torch.Tensor],
        gen_state: torch.Tensor,
        gen_next_state: torch.Tensor,
        gen_action: Optional[torch.Tensor],
        stats_dict: dict = {},
    ):
        """
        Computes the gradient penalty term for WGAN-GP using interpolation between
        expert and generated samples.
        """
        batch_size = gen_state.size(0)

        # Sample interpolation factors
        alpha_shape = [batch_size] + [1] * (expert_state.ndim - 1)
        alpha = torch.rand(alpha_shape, device=self.device)

        # Interpolation between expert and generated inputs
        expert_parts = [expert_state, expert_next_state]
        gen_parts = [gen_state, gen_next_state]
        if getattr(self._reward.cfg, "use_action", False):
            if expert_action is None:
                expert_action = torch.zeros(
                    *expert_state.shape[:-1],
                    self._reward.cfg.action_dim,
                    device=expert_state.device,
                    dtype=expert_state.dtype,
                )
            if gen_action is None:
                gen_action = torch.zeros(
                    *gen_state.shape[:-1],
                    self._reward.cfg.action_dim,
                    device=gen_state.device,
                    dtype=gen_state.dtype,
                )
            expert_parts.append(expert_action)
            gen_parts.append(gen_action)

        _expert_input = torch.cat(expert_parts, dim=-1)
        _gen_input = torch.cat(gen_parts, dim=-1)
        interpolated = torch.lerp(_expert_input, _gen_input, alpha).requires_grad_(True)

        # Reward model output on interpolated samples
        interpolated_scores = self._reward.model(interpolated)

        # Compute gradients w.r.t. interpolated inputs
        gp_gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gp_gradients = gp_gradients.view(batch_size, -1)
        gp_gradient_norm = gp_gradients.norm(2, dim=1)
        penalty = ((gp_gradient_norm - 1.0) ** 2).mean()
        stats_dict['gp_gradient_mean'] = gp_gradient_norm.mean().item()

        return penalty

    def update_value( # TODO: share policy-planned trajectories with policy update
        self,
        latent_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_latent_batch: torch.Tensor,
        learner_cfg : 'ValueLearnerCfg',
        terminal_mask: Optional[torch.Tensor] = None,
    ):
        """
        Update value function using TD learning.

        Args:
            latent_batch: [batch, latent_dim] - current latent states
            action_batch: [batch, action_dim] - actions taken
            next_latent_batch: [batch, latent_dim] - next latent states
            learner_cfg: Value learner configuration
            terminal_mask: Optional [batch] boolean tensor indicating terminal transitions.
                          If True, the next state value should be 0 (episode ended).
        """

        with torch.no_grad():
            rewards_batch = self._reward(latent_batch, next_latent_batch, action=action_batch)
            rewards_batch = self._calibrate_rewards(rewards_batch)
            _next_plans = self._policy.plan(next_latent_batch) # [batch_size, H, action_dim]
            pred_next_latent_batch_traj = self._dynamics(
                z0=next_latent_batch, controls=_next_plans,
            ) # [batch_size, H+1, latent_dim]
            next_vals = self._n_step_return_lambda(
                z_traj=pred_next_latent_batch_traj[..., :-1, :],
                actions_traj=_next_plans,
                next_z_traj=pred_next_latent_batch_traj[..., 1:, :],
                lam=learner_cfg.lam,
                value=self._value_target,
                reward_fn=self._reward,
            )

            # Apply terminal mask: V(next_state) = 0 for terminal transitions
            # This implements the standard RL (1 - done) * V(s') term
            if terminal_mask is not None:
                next_vals = next_vals * (~terminal_mask).float()

            td_target = rewards_batch + learner_cfg.gamma * next_vals

        # Use normalized states for value function
        all_q_values = self._value(latent_batch, action_batch, return_type='all')  # [num_q, batch]

        loss = sum(F.mse_loss(q_vals, td_target) for q_vals in all_q_values) / len(all_q_values)

        self._value_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._value.parameters(), self.value_learner_cfg.max_grad_norm
        )
        self._value_opt.step()

        self._mean_stats['Value/td_target_mean'] += td_target.mean().item()
        self._mean_stats['Value/mean_q_value'] += all_q_values.mean().item()

        # Update target networks using Polyak averaging
        self._soft_update_target_value()

        return loss.item()

    def update_policy(
        self,
        latent_batch: torch.Tensor,
    ): # [batch_size, H, state_dim]

        self._value.track_q_grad(enable=False)

        with torch.no_grad():
            old_policy_out = self._policy.model(latent_batch)
            old_mu, _old_log_std = torch.chunk(old_policy_out, 2, dim=-1)
            old_mu = torch.tanh(
                old_mu.view(latent_batch.shape[0], self._policy.horizon, self._policy.action_dim)[:, 0]
            )

        plans = self._policy.plan(latent_batch)
        log_prob_plans = self._policy._log_prob_plans # [batch_size, H, 1]
        pred_latents = self._dynamics(z0=latent_batch, controls=plans)

        alpha = self._log_alpha.exp().squeeze()

        # Multi-step model-based TDLambda
        qs = self._n_step_return_lambda( # [batch_size]
            z_traj=pred_latents[..., :-1, :],
            actions_traj=plans,
            next_z_traj=pred_latents[..., 1:, :],
            value=self._value,
            lam=self.policy_learner_cfg.lam,
            reward_fn=self._reward,
            log_probs=log_prob_plans.squeeze(-1),
            alpha=alpha.detach(),
        )
        mpail_policy_loss = -qs.mean()
        loss = self.cfg.mpail_policy_loss_coeff * mpail_policy_loss
        bc_loss = None
        if self._bc_obs is not None and self._bc_actions is not None and self.cfg.demo_action_bc_coeff > 0:
            bc_batch_size = min(latent_batch.shape[0], self._bc_obs.shape[0])
            bc_idx = torch.randint(self._bc_obs.shape[0], (bc_batch_size,), device=self.device)
            bc_obs = {"policy": self._bc_obs[bc_idx]}
            if self._obs_normalizer is not None:
                bc_obs = self._obs_normalizer(bc_obs)
            bc_z = self._encoder(bc_obs)
            bc_out = self._policy.model(bc_z)
            bc_mu, _bc_log_std = torch.chunk(bc_out, 2, dim=-1)
            bc_mu = torch.tanh(bc_mu.view(bc_batch_size, self._policy.horizon, self._policy.action_dim)[:, 0])
            bc_target = self._bc_actions[bc_idx]
            bc_loss = F.mse_loss(bc_mu, bc_target)
            loss = loss + self.cfg.demo_action_bc_coeff * bc_loss

        trust_region_loss = None
        if self.cfg.policy_trust_region_coeff > 0:
            new_policy_out = self._policy.model(latent_batch)
            new_mu, _new_log_std = torch.chunk(new_policy_out, 2, dim=-1)
            new_mu = torch.tanh(
                new_mu.view(latent_batch.shape[0], self._policy.horizon, self._policy.action_dim)[:, 0]
            )
            trust_region_loss = F.mse_loss(new_mu, old_mu)
            loss = loss + self.cfg.policy_trust_region_coeff * trust_region_loss

        self._policy_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._policy.parameters(), self.policy_learner_cfg.max_grad_norm
        )
        self._policy_opt.step()

        # Update alpha
        self._alpha_opt.zero_grad()
        alpha_loss = alpha * (-log_prob_plans.mean(-2).detach() - self._target_entropy).mean()
        alpha_loss.backward()
        self._alpha_opt.step()

        # Log stats
        self._mean_stats['Policy/alpha'] += alpha.item()
        self._mean_stats['Policy/alpha_loss'] += alpha_loss.item()
        self._mean_stats['Policy/entropy'] += -log_prob_plans.mean().item()
        self._mean_stats['Policy/mean_std'] += self._policy.std.mean().detach().item()
        self._mean_stats['Policy/mpail_loss'] += mpail_policy_loss.item()
        self._mean_stats['Policy/mpail_loss_coeff'] += self.cfg.mpail_policy_loss_coeff
        if bc_loss is not None:
            self._mean_stats['Policy/bc_loss'] += bc_loss.item()
        if trust_region_loss is not None:
            self._mean_stats['Policy/trust_region_loss'] += trust_region_loss.item()
            self._mean_stats['Policy/trust_region_coeff'] += self.cfg.policy_trust_region_coeff

        self._value.track_q_grad(enable=True)
        return loss.item()

    def _n_step_return_lambda(
            self, z_traj, actions_traj, next_z_traj, value, lam: float, reward_fn,
            alpha:float=None, log_probs:torch.Tensor=None,
            value_return_type: str='min'
        ):
        '''
        Computes the n-step lambda return. Defined recursively as:
        G_t = lam * Q(z_t, a_t) + (1 - lam) * [r_t + gamma * G_{t+1}]
        Tensors are shape [batch, H, dim]
        Args:
            z_traj: Latent trajectory [batch, H, latent_dim]
            actions_traj: Action trajectory [batch, H, action_dim]
            next_z_traj: Next latent trajectory [batch, H, latent_dim]
            value: Value function to use for bootstrapping
            lam: Lambda parameter for return computation
            reward_fn: Reward function to use
            alpha: Entropy regularization coefficient
            log_probs: Log probabilities of actions [batch, H] (for entropy regularization)
            value_return_type: Return type for value function ('min', 'mean', 'all')
        Returns:
            _return: Lambda return [batch]
        '''

        _gam, _H = self.value_learner_cfg.gamma, self.cfg.loss_horizon
        _all_rewards = reward_fn( # [batch, H]
            z_traj, next_z_traj, action=actions_traj
        )
        _all_rewards = self._calibrate_rewards(_all_rewards)
        _all_vals = value(z_traj, actions_traj, return_type=value_return_type)  # [batch, H] or [num_q, batch, H]
        if log_probs is not None:
            # Entropy regularization adjustment
            _all_rewards = _all_rewards - alpha * log_probs
            _all_vals = _all_vals - alpha * log_probs

        steps = torch.arange(_H - 1, device=self.device, dtype=self.dtype)
        W = (_gam * lam) ** steps

        b = _all_rewards[..., :-1] + _gam * (1 - lam) * _all_vals[..., 1:]
        b[..., -1] += _gam * lam * _all_vals[..., -1]

        _lambda_return = b @ W.view(-1, 1)

        _return = lam * _all_vals[..., 0] + (1 - lam) * _lambda_return.squeeze(-1)

        return _return
