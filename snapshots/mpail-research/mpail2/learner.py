import copy
import math
import time
import torch
from collections import defaultdict
import torch.nn.functional as F
from typing import TYPE_CHECKING, Dict, Optional

from mpail2.encoder import Coder
from mpail2.dynamics import Dynamics, SIGReg
from mpail2.reward import Reward
from mpail2.value import EnsembleValue
from mpail2.sampling import PolicyNetwork
from mpail2.planner import Planner
from .storage import RolloutStorage as Storage, DictDemoDataset
from .utils import resolve_optim, compute_disc_train_stats
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
        self._value:EnsembleValue = self.planner.value
        self._policy:PolicyNetwork = self.planner.sampling.policy

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
        self._sigreg = None
        if self.dynamics_learner_cfg.sigreg_coeff > 0.0:
            self._sigreg = SIGReg(
                knots=self.dynamics_learner_cfg.sigreg_knots,
                num_proj=self.dynamics_learner_cfg.sigreg_num_proj,
            ).to(device=self.device, dtype=self.dtype)

        #
        # POLICY SAMPLING SETUP
        #

        self.policy_learner_cfg = self.cfg.policy_learner_cfg
        self._policy_opt = resolve_optim(self.policy_learner_cfg.opt)(
            self._policy.parameters(),
            **self.policy_learner_cfg.opt_params
        )
        self._policy_target = self._create_target_policy_network()

        # Initialize temperature as a learnable parameter
        # self._log_temp = torch.nn.Parameter(torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self._log_temp = self.planner._log_temp
        self._log_iq_alpha = torch.tensor(
            math.log(1.0), requires_grad=True, device=self.device, dtype=self.dtype
        )
        self._temp_opt = resolve_optim(self.policy_learner_cfg.opt)(
            [self._log_iq_alpha], lr=self.policy_learner_cfg.temp_lr
        )
        self._iq_alpha_opt = resolve_optim(self.policy_learner_cfg.opt)(
            [self._log_iq_alpha], lr=self.policy_learner_cfg.temp_lr
        )
        self._target_entropy = self.policy_learner_cfg.target_entropy_per_action_dim * self._policy.action_dim
        self._policy_target_tau = self.policy_learner_cfg.intrinsic_polyak_tau
        self._intrinsic_norm_eps = self.planner.cfg.intrinsic_norm_eps

        # Intrinsic exploration coefficients are stored on the planner so
        # planning/learning share them. Three per-source learnable Lagrangian
        # duals scale the dyn / rew / val disagreement bonuses.
        _lambda_params = [
            self.planner._log_lambda_intrinsic_dyn,
            self.planner._log_lambda_intrinsic_rew,
            self.planner._log_lambda_intrinsic_val,
        ]
        self._lambda_opt = torch.optim.Adam(
            _lambda_params,
            lr=self.policy_learner_cfg.intrinsic_lambda_lr,
        )

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
            )

            self.planner.set_obs_normalizer(self._obs_normalizer)

        self._iql_obs = None
        self._iql_next_obs = None
        self._iql_actions = None
        self._iql_not_done = None

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
        self._demo_dataset = DictDemoDataset(
            self._demonstrations,
            horizon=self.cfg.loss_horizon,
            device=self.device,
            obs_normalizer=self._obs_normalizer,
        )

    @staticmethod
    def _compact_omnireset_policy_obs(
        policy_obs: torch.Tensor,
        *,
        include_ee_to_hole: bool = True,
        include_ee_vel: bool = False,
    ) -> torch.Tensor:
        if policy_obs.shape[-1] != 215:
            raise ValueError(f"Expected OmniReset policy obs dim 215, got {policy_obs.shape[-1]}")
        peg_in_hole = policy_obs[..., 24:30]
        joint_pos_latest = policy_obs[..., 113:125]
        ee_pose_prev = policy_obs[..., 143:149]
        ee_pose = policy_obs[..., 149:155]
        peg_pose = policy_obs[..., 179:185]
        ee_to_peg_xyz = peg_pose[..., :3] - ee_pose[..., :3]
        ee_to_peg_rot = peg_pose[..., 3:6] - ee_pose[..., 3:6]
        gripper_mean = joint_pos_latest[..., 6:12].mean(dim=-1, keepdim=True)
        parts = [peg_in_hole, ee_to_peg_xyz, ee_to_peg_rot]
        if include_ee_to_hole:
            hole_pose = policy_obs[..., 209:215]
            parts.extend([hole_pose[..., :3] - ee_pose[..., :3], hole_pose[..., 3:6] - ee_pose[..., 3:6]])
        parts.append(gripper_mean)
        if include_ee_vel:
            parts.append(ee_pose[..., :3] - ee_pose_prev[..., :3])
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _planner_v2_omnireset_policy_obs(policy_obs: torch.Tensor) -> torch.Tensor:
        if policy_obs.shape[-1] != 215:
            raise ValueError(f"Expected OmniReset policy obs dim 215, got {policy_obs.shape[-1]}")
        peg_in_hole_prev = policy_obs[..., 18:24]
        peg_in_hole = policy_obs[..., 24:30]
        joint_pos_latest = policy_obs[..., 113:125]
        ee_pose_prev = policy_obs[..., 143:149]
        ee_pose = policy_obs[..., 149:155]
        peg_pose_prev = policy_obs[..., 173:179]
        peg_pose = policy_obs[..., 179:185]
        hole_pose = policy_obs[..., 209:215]
        return torch.cat(
            [
                peg_in_hole,
                peg_in_hole - peg_in_hole_prev,
                ee_pose,
                peg_pose,
                hole_pose,
                peg_pose - ee_pose,
                hole_pose - ee_pose,
                joint_pos_latest[..., 6:12],
                ee_pose - ee_pose_prev,
                peg_pose - peg_pose_prev,
            ],
            dim=-1,
        )

    @classmethod
    def _planner_v3_omnireset_policy_obs(cls, policy_obs: torch.Tensor) -> torch.Tensor:
        v2 = cls._planner_v2_omnireset_policy_obs(policy_obs)
        prev_actions = policy_obs[..., 30:65]
        joint_pos_prev = policy_obs[..., 101:113]
        joint_pos = policy_obs[..., 113:125]
        return torch.cat([v2, prev_actions, joint_pos, joint_pos - joint_pos_prev], dim=-1)

    def _adapt_demo_policy_obs(self, obs: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_obs_dim = next(iter(self._demonstrations.values())).shape[-1]
        if obs.shape[-1] == target_obs_dim:
            return obs, next_obs
        if obs.shape[-1] == 215 and target_obs_dim in (13, 19, 22, 60, 119):
            if target_obs_dim == 60:
                return self._planner_v2_omnireset_policy_obs(obs), self._planner_v2_omnireset_policy_obs(next_obs)
            if target_obs_dim == 119:
                return self._planner_v3_omnireset_policy_obs(obs), self._planner_v3_omnireset_policy_obs(next_obs)
            transform_kwargs = {
                "include_ee_to_hole": target_obs_dim in (19, 22),
                "include_ee_vel": target_obs_dim == 22,
            }
            return (
                self._compact_omnireset_policy_obs(obs, **transform_kwargs),
                self._compact_omnireset_policy_obs(next_obs, **transform_kwargs),
            )
        raise ValueError(f"Cannot adapt demo obs dim {obs.shape[-1]} to learner obs dim {target_obs_dim}")

    def _normalize_demo_actions(self, actions: torch.Tensor) -> torch.Tensor:
        raw_limit = getattr(self.cfg, "demo_action_raw_limit", None)
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
        print(
            f"[MPAIL2Learner] Loaded IQ-Learn demo transitions: {self._iql_obs.shape[0]} samples",
            flush=True,
        )

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
        return self._log_iq_alpha.exp().squeeze()

    @staticmethod
    def _clip_grad_norm_if_set(parameters, max_norm):
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)

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
                self._log_iq_alpha.copy_(
                    torch.tensor(math.log(float(init_temp)), device=self.device, dtype=self._log_iq_alpha.dtype)
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

            z_dyn = self._encoder(expert_obs, return_type="avg")
            with torch.no_grad():
                next_z_dyn = self._encoder(expert_next_obs, return_type="avg").detach()
            if dyn_coeff > 0.0:
                pred_next_z = self._dynamics(z0=z_dyn, controls=expert_actions.unsqueeze(1))[:, 1, :]
                dyn_loss = F.mse_loss(pred_next_z, next_z_dyn)
                self._dyn_opt.zero_grad(set_to_none=True)
                (dyn_coeff * dyn_loss).backward()
                self._clip_grad_norm_if_set(self._encoder.parameters(), self.policy_learner_cfg.max_grad_norm)
                self._clip_grad_norm_if_set(self._dynamics.parameters(), self.value_learner_cfg.max_grad_norm)
                self._dyn_opt.step()
            else:
                dyn_loss = torch.zeros((), device=self.device, dtype=self.dtype)

            expert_z = self._encoder(expert_obs, return_type="avg")
            with torch.no_grad():
                expert_next_z = self._encoder(expert_next_obs, return_type="avg").detach()

            if policy_batch_size > 0:
                policy_idx = torch.randint(self._iql_obs.shape[0], (policy_batch_size,), device=self.device)
                raw_policy_obs = {"policy": self._iql_obs[policy_idx]}
                policy_obs = self._obs_normalizer(raw_policy_obs) if self._obs_normalizer is not None else raw_policy_obs
                policy_z = self._encoder(policy_obs, return_type="avg")
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
            self._clip_grad_norm_if_set(self._encoder.parameters(), self.policy_learner_cfg.max_grad_norm)
            self._clip_grad_norm_if_set(self._value.parameters(), self.value_learner_cfg.max_grad_norm)
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
            self._clip_grad_norm_if_set(self._policy.parameters(), self.policy_learner_cfg.max_grad_norm)
            self._policy_opt.step()

            alpha_loss = torch.zeros((), device=self.device, dtype=self.dtype)
            if learn_temp:
                self._iq_alpha_opt.zero_grad(set_to_none=True)
                alpha_loss = (
                    self._iqlearn_alpha() * (-policy_logp.detach() - self._target_entropy)
                ).mean()
                alpha_loss.backward()
                self._iq_alpha_opt.step()

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
                    "expert_iq_reward_mean": float(expert_iq_reward.mean().item()),
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
            "[MPAIL2Learner] IQ-Learn pretrained same MPAIL3 models: "
            f"steps={int(steps)} expert_batch_size={batch_size} policy_batch_size={policy_batch_size} "
            f"dyn_loss={last_stats.get('dyn_loss', 0.0):.6f} "
            f"q_loss={last_stats.get('q_loss', 0.0):.6f} "
            f"actor_loss={last_stats.get('actor_loss', 0.0):.6f} "
            f"bc_loss={last_stats.get('bc_loss', 0.0):.6f} "
            f"alpha={last_stats.get('alpha', 0.0):.6f} "
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
                z = self._encoder(obs, return_type="avg").detach()
                next_z = self._encoder(next_obs, return_type="avg").detach()
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
            targets = torch.cat([expert_target, policy_target, random_target], dim=0)
            pred = self._reward(states, next_states, return_type="avg")
            loss = F.mse_loss(pred, targets)
            self._reward_opt.zero_grad(set_to_none=True)
            loss.backward()
            self._clip_grad_norm_if_set(self._reward.parameters(), getattr(self.reward_learner_cfg, "max_grad_norm", None))
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
            "[MPAIL2Learner] Distilled IQ-Learn recovered reward into MPAIL3 reward model: "
            f"steps={int(steps)} batch_size={batch_size} "
            f"loss={last_stats.get('loss', 0.0):.6f} "
            f"target_mean={last_stats.get('target_mean', 0.0):.6f} "
            f"pred_mean={last_stats.get('pred_mean', 0.0):.6f}",
            flush=True,
        )

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
        obs = {"policy": self._iql_obs}
        next_obs = {"policy": self._iql_next_obs}
        added = self.storage.add_demo_transitions_to_replay(
            obs=obs,
            next_obs=next_obs,
            actions=self._iql_actions,
            not_done=self._iql_not_done,
            max_trajectories=max_trajs,
        )
        print(
            "[MPAIL2Learner] Seeded expert demos into replay: "
            f"trajs={added}/{max_trajs} transitions={self._iql_actions.shape[0]} "
            f"protect={bool(getattr(self.cfg, 'protect_demo_success_replay', True))}",
            flush=True,
        )

    def eval(self):
        self.planner.eval()

    def train(self):
        self.planner.train()

    def _bc_policy_mean_action(self, obs) -> torch.Tensor:
        """Return the current policy's deterministic first action for diagnostics."""
        with torch.no_grad():
            z = self._encoder(obs, return_type="avg")
            return self._policy_mean_from_latent(z)

    def act(self, obs, critic_obs=None, vis_rollouts=False):
        '''Performs forward pass on MPPI and collects discriminator logits for MPAIL
        TODO: Critic obs?'''

        # Do NOT normalize obs for planner - planner uses raw observations
        # Normalization is only applied when feeding to value function

        # Compute the actions and values
        if critic_obs is None:
            critic_obs = obs

        with torch.no_grad():
            actions = self.planner.act(obs, vis_rollouts=vis_rollouts, deterministic=not self.planner.cfg.random_act)

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

    def _create_target_policy_network(self):
        target_network = copy.deepcopy(self._policy)
        for param in target_network.parameters():
            param.requires_grad_(False)
        target_network.eval()
        for target_param, main_param in zip(target_network.parameters(), self._policy.parameters()):
            target_param.data.copy_(main_param.data)
        return target_network

    def _soft_update_target_value(self):
        """Soft update using Polyak averaging: target = (1-tau) * target + tau * main"""
        for target_param, main_param in zip(self._value_target.parameters(), self._value.parameters()):
            target_param.data.copy_(
                (1.0 - self._polyak_tau) * target_param.data + self._polyak_tau * main_param.data
            )

    def _soft_update_target_policy(self):
        for target_param, main_param in zip(self._policy_target.parameters(), self._policy.parameters()):
            target_param.data.copy_(
                (1.0 - self._policy_target_tau) * target_param.data + self._policy_target_tau * main_param.data
            )

    def update(self, iteration: int) -> Dict[str, float]:

        self.storage.process_replay_buffer() # Move episode storage to replay buffer

        _num_flat_trajs = self.storage.replay_batch_size // self.cfg.loss_horizon
        demo_batch_size = _num_flat_trajs  # Number of subtrajectories to sample

        # Calculate number of updates based on replay ratio and new data collected
        num_updates = max(1, int(self.replay_ratio * self._total_batch_size))

        # Use custom batch generator to avoid duplicate sampling
        demo_generator = self._demo_dataset.mini_traj_batch_generator(
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

            # (1) UPDATE DYNAMICS
            _start = time.time()
            self._mean_stats['Dyn/mean_loss'] += self.update_dynamics( # TODO: dont duplicate encoding
                obs_batch_traj=replay_obs_traj,
                next_obs_batch_traj=replay_next_obs_traj,
                actions_batch_traj=replay_actions_traj,
                # demo_obs_batch_traj=demo_obs_traj,
            )
            self._tot_stats['Dyn/learn_time'] += time.time() - _start

            # Encode observations for downstream heads (optionally slower target encoder)
            with torch.no_grad():
                replay_latent_traj = self._encoder(replay_obs_traj, return_type='avg')
                replay_next_latent_traj = self._encoder(replay_next_obs_traj, return_type='avg')
                demo_latent_flat = self._encoder(demo_obs_flat, return_type='avg')
                demo_next_latent_flat = self._encoder(demo_next_obs_flat, return_type='avg')

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

            # (2) UPDATE REWARD
            _start = time.time()
            if getattr(self.cfg, "skip_reward_updates", False):
                self._mean_stats['Reward/skipped'] += 1.0
            else:
                _reward_stats = self.update_reward(
                    demo_latent_batch=demo_latent_flat,
                    demo_next_latent_batch=demo_next_latent_flat,
                    latent_batch=latent_batch_flat,
                    next_latent_batch=next_latent_batch_flat,
                    gen_negative_mask=None if replay_is_demo_flat is None else ~replay_is_demo_flat,
                )
                for key, value in _reward_stats.items():
                    self._mean_stats[key] += value
            self._tot_stats['Reward/learn_time'] += time.time() - _start

            # (3) UPDATE VALUE
            _start = time.time()
            self._mean_stats['Value/mean_loss'] += self.update_value(
                latent_batch=latent_batch_flat,
                action_batch=actions_batch_flat,
                next_latent_batch=next_latent_batch_flat,
                learner_cfg=self.value_learner_cfg,
                terminal_mask=terminal_mask_flat,
            )
            self._tot_stats['Value/learn_time'] += time.time() - _start

            # (4) UPDATE POLICY
            _start = time.time()
            self._mean_stats['Policy/mean_loss'] += self.update_policy( # TODO: dont duplicate encoding
                latent_batch=latent_batch_flat,
            )
            self._tot_stats['Policy/learn_time'] += time.time() - _start

            actual_updates += 1
            self._mean_stats['Data/mean_latent_norm'] += latent_batch_flat.norm(dim=-1).mean().item()

        # Average stats over updates
        for key in self._mean_stats.keys():
            self._mean_stats[key] /= max(1, actual_updates)

        # Store training info
        self._tot_stats['Training/num_updates'] = actual_updates
        self._tot_stats['Training/replay_ratio'] = self.replay_ratio

        # Log how much data is in the replay buffer
        self._tot_stats['Storage/replay_size'] = self.storage.num_trajectories_stored * self.storage.num_steps_per_env
        for key, value in self.storage.compute_replay_stats().items():
            self._tot_stats[key] = value

        # -- Clear the storage
        self.storage.clear()

        self._mean_stats.update(self._tot_stats)

        return self._mean_stats

    def update_dynamics(
        self,
        obs_batch_traj: torch.Tensor,
        next_obs_batch_traj: torch.Tensor, # For optional decoding loss
        actions_batch_traj: torch.Tensor,
    ):

        # Encode only z0 from obs_batch_traj to avoid unnecessary full-trajectory encoding.
        # time_dim = actions_batch_traj.ndim - 2
        # obs0_batch = {k: v.select(dim=time_dim, index=0) for k, v in obs_batch_traj.items()}
        # z0 = self._encoder(obs0_batch)
        latent_batch_traj = self._encoder(obs_batch_traj)
        z0 = latent_batch_traj[..., 0, :]

        next_latent_batch_traj = self._encoder(next_obs_batch_traj)

        _H = actions_batch_traj.shape[-2]
        rho = self.dynamics_learner_cfg.rho
        _rhos = rho ** torch.arange(_H, device=self.device)

        pred_latents = self._dynamics( # [batch_size, H+1, latent_dim]
            z0=z0, controls=actions_batch_traj, return_type='all'
        )

        # JEP loss: L = sum_t rho^t || z_{t+1} - f(z_t, a_t) ||_2^2
        _jep_se = (pred_latents[..., 1:, :] - next_latent_batch_traj).pow(2)
        jep_loss = (_rhos[..., None] * _jep_se).mean()

        # SIGReg loss: expects [H, B, D]
        sigreg_loss = self._sigreg(next_latent_batch_traj.transpose(0, 1))

        loss = jep_loss + self.dynamics_learner_cfg.sigreg_coeff * sigreg_loss

        # Decoder for visualization if specified
        if self._decoder:
            _recon_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            obs_key = self._decoder.obs_key
            if obs_key in next_obs_batch_traj:
                next_latent_for_decoder = (
                    next_latent_batch_traj.detach().mean(dim=0)
                    if next_latent_batch_traj.ndim == 4
                    else next_latent_batch_traj.detach()
                )
                _recon_loss = F.mse_loss(
                    self._decoder(next_latent_for_decoder),
                    next_obs_batch_traj[obs_key],
                )
        else:
            _recon_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        self._mean_stats['Dyn/jep_loss'] += jep_loss.item()
        self._mean_stats['Dyn/recon_loss'] += _recon_loss.item()
        self._mean_stats['Dyn/sigreg_loss'] += sigreg_loss.item()

        self._dyn_opt.zero_grad()
        loss.backward()
        if self._decoder:
            _recon_loss.backward() # Backprop through decoder
        self._dyn_opt.step()

        return loss.item()

    def update_reward(
        self,
        latent_batch: torch.Tensor,
        next_latent_batch: torch.Tensor,
        demo_latent_batch: torch.Tensor,
        demo_next_latent_batch: torch.Tensor,
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
            else:
                return {
                    "Reward/skipped_all_demo_replay": 1.0,
                    "Reward/replay_demo_success_fraction": replay_demo_fraction,
                }

        if demo_latent_batch.shape[0] != latent_batch.shape[0]:
            demo_idx = torch.randint(demo_latent_batch.shape[0], (latent_batch.shape[0],), device=self.device)
            demo_latent_batch = demo_latent_batch[demo_idx]
            demo_next_latent_batch = demo_next_latent_batch[demo_idx]

        _demo_rewards = self._reward(demo_latent_batch, demo_next_latent_batch, return_type='all')
        _gen_rewards = self._reward(latent_batch, next_latent_batch, return_type='all')

        # WGAN loss: member-wise objectives averaged across ensemble heads
        loss = (-_demo_rewards.mean(dim=-1) + _gen_rewards.mean(dim=-1)).mean()

        train_stats = {}
        # add gradient penalty if coefficient is specified
        gp_loss = self.reward_learner_cfg.gp_coeff * self._compute_gradient_penalty(
            expert_state=demo_latent_batch.detach(),
            expert_next_state=demo_next_latent_batch.detach(),
            gen_state=latent_batch.detach(),
            gen_next_state=next_latent_batch.detach(),
            stats_dict=train_stats,
        )
        loss += gp_loss

        with torch.no_grad():
            # FOR STATS LOGGING
            _output = torch.cat((
                _demo_rewards.reshape(-1),
                _gen_rewards.reshape(-1)
            ), dim=0)
            _target = torch.cat(( # [ 1 \\ 0 ]
                torch.ones_like(_demo_rewards.reshape(-1)),
                torch.zeros_like(_gen_rewards.reshape(-1))
            ), dim=0)

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
                "demo_std": _demo_rewards.reshape(-1).std().item(),
                "gen_std": _gen_rewards.reshape(-1).std().item(),
                "replay_demo_success_fraction": replay_demo_fraction,
            })

            if self.reward_learner_cfg.gp_coeff is not None and self.reward_learner_cfg.gp_coeff > 0:
                train_stats["gp_loss"] = gp_loss.item()

        _train_stats = {}
        for key, value in train_stats.items():
            _train_stats[f"Reward/{key}"] = value

        _train_stats['Reward/learner_latent_std'] = latent_batch.std(-2).mean().item()
        _train_stats['Reward/demo_latent_std'] = demo_latent_batch.std(-2).mean().item()

        return _train_stats

    def _compute_gradient_penalty(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        gen_state: torch.Tensor,
        gen_next_state: torch.Tensor,
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
        _expert_input = torch.cat((expert_state, expert_next_state), dim=-1)
        _gen_input = torch.cat((gen_state, gen_next_state), dim=-1)
        interpolated = torch.lerp(_expert_input, _gen_input, alpha).requires_grad_(True)

        interpolated_scores = self._reward.forward_from_concat(interpolated, return_type='all')
        if interpolated_scores.ndim == 1:
            interpolated_scores = interpolated_scores.unsqueeze(0)

        penalties = []
        grad_norms = []
        for member_idx in range(interpolated_scores.shape[0]):
            member_scores = interpolated_scores[member_idx]
            gp_gradients = torch.autograd.grad(
                outputs=member_scores,
                inputs=interpolated,
                grad_outputs=torch.ones_like(member_scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gp_gradients = gp_gradients.view(batch_size, -1)
            gp_gradient_norm = gp_gradients.norm(2, dim=1)
            penalties.append(((gp_gradient_norm - 1.0) ** 2).mean())
            grad_norms.append(gp_gradient_norm.mean())

        penalty = torch.stack(penalties).mean()
        stats_dict['gp_gradient_mean'] = torch.stack(grad_norms).mean().item()

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

            rewards_batch = self._reward(latent_batch, next_latent_batch, action=action_batch, return_type='min')
            _next_plans = self._policy.plan(next_latent_batch) # [batch_size, H, action_dim]
            _next_log_prob_plans = self._policy._log_prob_plans.squeeze(-1)

            # Roll out from s' using the policy's plan: [s', s'', ..., s'^H]
            _next_pred_latents_all = self._dynamics(
                z0=next_latent_batch, controls=_next_plans, return_type='all'
            ) # [E, batch_size, H+1, latent_dim]
            _next_pred_latents = (
                _next_pred_latents_all.mean(dim=0)
                if self._dynamics.ensemble_size > 1
                else _next_pred_latents_all
            )

            # Intrinsic rewards at (s'+t, a'+t) for t=0..H-1 — i.e., I_u(s',a') is
            # the first term, matching the paper's bootstrap y = r + γ[Q̄ − α1 log π + α2 I_u(s',a')]
            _next_intrinsic_rewards = self.planner._compute_intrinsic_reward(
                pred_latents_all=_next_pred_latents_all,
                rollouts=_next_pred_latents,
                actions=_next_plans,
            )

            next_vals = self._n_step_return_lambda(
                z_traj=_next_pred_latents[..., :-1, :],
                actions_traj=_next_plans,
                next_z_traj=_next_pred_latents[..., 1:, :],
                lam=learner_cfg.lam,
                value=self._value_target,
                reward_fn=self._reward,
                intrinsic_reward=_next_intrinsic_rewards,
                log_probs=_next_log_prob_plans,
                # log_probs=None,
                temperature=self._iqlearn_alpha().detach(),
                return_type='min',
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
        if self.value_learner_cfg.max_grad_norm is not None:
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

        plans = self._policy.plan(latent_batch)
        log_prob_plans = self._policy._log_prob_plans # [batch_size, H, 1]
        pred_latents_all = self._dynamics(z0=latent_batch, controls=plans, return_type='all')
        pred_latents = (
            pred_latents_all.mean(dim=0)
            if self._dynamics.ensemble_size > 1
            else pred_latents_all
        )

        temperature = self._iqlearn_alpha()

        intrinsic_components = self.planner.compute_intrinsic_components(
            pred_latents_all=pred_latents_all,
            rollouts=pred_latents,
            actions=plans,
        )
        intrinsic_reward = self.planner._compute_intrinsic_reward(
            pred_latents_all=pred_latents_all,
            rollouts=pred_latents,
            actions=plans,
        )
        if not self.policy_learner_cfg.use_intrinsic_bonus:
            intrinsic_reward = torch.zeros_like(intrinsic_reward)

        # Multi-step model-based TDLambda
        qs = self._n_step_return_lambda( # [batch_size]
            z_traj=pred_latents[..., :-1, :],
            actions_traj=plans,
            next_z_traj=pred_latents[..., 1:, :],
            value=self._value,
            lam=self.policy_learner_cfg.lam,
            reward_fn=self._reward,
            log_probs=log_prob_plans.squeeze(-1),
            temperature=temperature.detach(),
            intrinsic_reward=intrinsic_reward if self.policy_learner_cfg.use_intrinsic_bonus else None,
        )
        loss = -qs.mean()

        self._policy_opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.policy_learner_cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
            self._policy.parameters(), self.policy_learner_cfg.max_grad_norm
        )
        self._policy_opt.step()

        # Update temperature
        self._temp_opt.zero_grad()
        temp_loss = temperature * (-log_prob_plans.mean(-2).detach() - self._target_entropy).mean()
        temp_loss.backward()
        self._temp_opt.step()

        sigma_pi_dyn = intrinsic_components["dyn"].detach().mean()
        sigma_pi_rew = intrinsic_components["rew"].detach().mean()
        sigma_pi_val = intrinsic_components["val"].detach().mean()
        sigma_target_dyn = torch.zeros_like(sigma_pi_dyn)
        sigma_target_rew = torch.zeros_like(sigma_pi_rew)
        sigma_target_val = torch.zeros_like(sigma_pi_val)
        lambda_loss_dyn = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        lambda_loss_rew = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        lambda_loss_val = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        lambda_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if self.policy_learner_cfg.use_intrinsic_bonus:
            with torch.no_grad():
                target_plans = self._policy_target.plan(latent_batch)
                target_pred_latents_all = self._dynamics(
                    z0=latent_batch, controls=target_plans, return_type='all'
                )
                target_pred_latents = (
                    target_pred_latents_all.mean(dim=0)
                    if self._dynamics.ensemble_size > 1
                    else target_pred_latents_all
                )
                target_components = self.planner.compute_intrinsic_components(
                    pred_latents_all=target_pred_latents_all,
                    rollouts=target_pred_latents,
                    actions=target_plans,
                )
                sigma_target_dyn = target_components["dyn"].mean()
                sigma_target_rew = target_components["rew"].mean()
                sigma_target_val = target_components["val"].mean()
            self._lambda_opt.zero_grad()
            lambda_loss_dyn = self.planner._log_lambda_intrinsic_dyn * (sigma_pi_dyn - sigma_target_dyn).detach()
            lambda_loss_rew = self.planner._log_lambda_intrinsic_rew * (sigma_pi_rew - sigma_target_rew).detach()
            lambda_loss_val = self.planner._log_lambda_intrinsic_val * (sigma_pi_val - sigma_target_val).detach()
            lambda_loss = lambda_loss_dyn + lambda_loss_rew + lambda_loss_val
            lambda_loss.backward()
            self._lambda_opt.step()

        self._soft_update_target_policy()

        # Log stats
        self._mean_stats['Policy/temperature'] += temperature.item()
        self._mean_stats['Policy/temp_loss'] += temp_loss.item()
        self._mean_stats['Policy/entropy'] += -log_prob_plans.mean().item()
        self._mean_stats['Policy/mean_std'] += self._policy.std.mean().detach().item()
        self._mean_stats['Policy/lambda_intrinsic_dyn'] += self.planner.lambda_intrinsic_dyn.item()
        self._mean_stats['Policy/lambda_intrinsic_rew'] += self.planner.lambda_intrinsic_rew.item()
        self._mean_stats['Policy/lambda_intrinsic_val'] += self.planner.lambda_intrinsic_val.item()
        self._mean_stats['Policy/intrinsic_reward_mean'] += intrinsic_reward.mean().detach().item()
        self._mean_stats['Policy/sigma_pi_dyn'] += sigma_pi_dyn.item()
        self._mean_stats['Policy/sigma_pi_rew'] += sigma_pi_rew.item()
        self._mean_stats['Policy/sigma_pi_val'] += sigma_pi_val.item()
        self._mean_stats['Policy/sigma_target_dyn'] += sigma_target_dyn.item()
        self._mean_stats['Policy/sigma_target_rew'] += sigma_target_rew.item()
        self._mean_stats['Policy/sigma_target_val'] += sigma_target_val.item()
        self._mean_stats['Policy/lambda_loss_dyn'] += lambda_loss_dyn.item()
        self._mean_stats['Policy/lambda_loss_rew'] += lambda_loss_rew.item()
        self._mean_stats['Policy/lambda_loss_val'] += lambda_loss_val.item()
        self._mean_stats['Policy/lambda_loss'] += lambda_loss.item()

        self._value.track_q_grad(enable=True)
        return loss.item()

    def _n_step_return_lambda(
            self, z_traj, actions_traj, next_z_traj, value, lam: float, reward_fn,
            temperature:float=None, log_probs:torch.Tensor=None,
            return_type: str='min',
            intrinsic_reward: Optional[torch.Tensor] = None,
        ):
        '''
        Computes the n-step lambda return. Defined recursively as:

        G_t = lam * (Q(z_t, a_t) + I_u - temp * logpi) + (1 - lam) * [r_t + gamma * G_{t+1}]

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
            z_traj, next_z_traj, action=actions_traj, return_type=return_type
        )
        _all_vals = value(z_traj, actions_traj, return_type=return_type)  # [batch, H] or [num_q, batch, H]

        if intrinsic_reward is not None:
            _all_rewards += intrinsic_reward
            _all_vals += intrinsic_reward

        if log_probs is not None:
            _all_rewards += -temperature * log_probs
            _all_vals += -temperature * log_probs

        steps = torch.arange(_H - 1, device=self.device, dtype=self.dtype)
        W = (_gam * lam) ** steps

        b = _all_rewards[..., :-1] + _gam * (1 - lam) * _all_vals[..., 1:]
        b[..., -1] += _gam * lam * _all_vals[..., -1]

        _lambda_return = b @ W.view(-1, 1)

        _return = lam * _all_vals[..., 0] + (1 - lam) * _lambda_return.squeeze(-1)

        return _return
