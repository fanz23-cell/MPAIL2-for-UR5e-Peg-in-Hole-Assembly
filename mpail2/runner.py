import os
import json
import time
import torch
import statistics
import gymnasium as gym
import wandb # TODO: does this start wandb?
import imageio.v2 as imageio
import numpy as np
from collections import deque
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Dict

# from isaaclab.envs import ManagerBasedRLEnv # Ideally runner works for all env types

from .learner import MPAIL2Learner
from .utils import Stats, task_stats, RolloutsVideo

if TYPE_CHECKING:
    from .configs.cfgs import MPAIL2RunnerCfg

class MPAIL2Runner:

    def __init__(self,
        demonstrations: torch.Tensor | Dict[str, torch.Tensor],
        env: gym.Env,
        runner_cfg: 'MPAIL2RunnerCfg',
        device: str = "cuda",
        dtype=torch.float32
    ):
        self.env = env
        self.cfg = runner_cfg
        self.log_cfg = self.cfg.log_cfg
        self.num_envs = env.unwrapped.num_envs
        self.device = device
        self.dtype = dtype
        self.demonstrations = demonstrations

        if type(self.demonstrations) is dict:
            for key in self.demonstrations:
                self.demonstrations[key] = self.demonstrations[key].to(device=self.device, dtype=self.dtype)
                # We assume demonstrations are stored as (N, 2, *obs_shape)
                # where N is flattened trajectories and 2 is (obs, next_obs)
                assert self.demonstrations[key].dim() >= 3 and self.demonstrations[key].shape[1] == 2, \
                    f"Demonstrations for key {key} must have shape (N, 2, *obs_shape), got {self.demonstrations[key].shape}"
        else:
            self.demonstrations = self.demonstrations.to(device=self.device, dtype=self.dtype)
            # We assume demonstrations are stored as (N, 2, *obs_shape)
            # where N is flattened trajectories and 2 is (obs, next_obs)
            assert self.demonstrations.dim() >= 3 and self.demonstrations.shape[1] == 2, \
                f"Demonstrations must have shape (N, 2, *obs_shape), got {self.demonstrations.shape}"
        self.demo_transition_count = self._count_demo_transitions(self.demonstrations)
        print(f"[INFO] Demonstration transitions: {self.demo_transition_count}")

        self.log_dir = self.log_cfg.log_dir
        self._num_steps_per_env = self.env.unwrapped.max_episode_length

        # Create MPAIL Learner
        self.learner = MPAIL2Learner(self.demonstrations, self.num_envs,
                                    self.cfg.learner_cfg, device=device)

        self.learner_cfg = self.cfg.learner_cfg
        self.action_scale = self._make_action_scale(getattr(self.cfg, "action_scale", None))
        self.teacher_policy = None
        self.teacher_bootstrap_steps = int(getattr(self.cfg, "teacher_bootstrap_steps", 0) or 0)
        self.teacher_blend_steps = int(getattr(self.cfg, "teacher_blend_steps", 0) or 0)
        teacher_policy_path = getattr(self.cfg, "teacher_policy_path", None)
        self.policy_control_after_online_teacher_bc = bool(
            getattr(self.cfg, "policy_control_after_online_teacher_bc", False)
        )
        self.teacher_diag_always = bool(getattr(self.cfg, "teacher_diag_always", False))
        self.teacher_dagger_collect = bool(getattr(self.cfg, "teacher_dagger_collect", False))
        self.teacher_dagger_buffer_size = int(getattr(self.cfg, "teacher_dagger_buffer_size", 0) or 0)
        self.teacher_dagger_export_path = getattr(self.cfg, "teacher_dagger_export_path", None)
        self.teacher_dagger_label_mode = getattr(self.cfg, "teacher_dagger_label_mode", "teacher")
        if self.teacher_dagger_label_mode not in ("teacher", "executed"):
            raise ValueError(f"Unknown teacher_dagger_label_mode={self.teacher_dagger_label_mode!r}")
        self._dagger_obs_buffer: torch.Tensor | None = None
        self._dagger_action_buffer: torch.Tensor | None = None
        self.teacher_intervention_cos_threshold = getattr(self.cfg, "teacher_intervention_cos_threshold", None)
        self.teacher_residual_beta = float(getattr(self.cfg, "teacher_residual_beta", 0.0) or 0.0)
        residual_final = getattr(self.cfg, "teacher_residual_final_beta", None)
        self.teacher_residual_final_beta = None if residual_final is None else float(residual_final)
        self.failure_strat_export_path = getattr(self.cfg, "failure_strat_export_path", None)
        self.failure_strat_label = str(getattr(self.cfg, "failure_strat_label", "policy") or "policy")
        if teacher_policy_path and (
            self.teacher_bootstrap_steps > 0
            or self.policy_control_after_online_teacher_bc
            or self.teacher_diag_always
            or self.teacher_dagger_collect
            or self.teacher_intervention_cos_threshold is not None
            or self.teacher_residual_beta > 0.0
            or (self.teacher_residual_final_beta is not None and self.teacher_residual_final_beta > 0.0)
        ):
            self.teacher_policy = torch.jit.load(teacher_policy_path, map_location=self.device).eval()

        # Extract observation shape from environment
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Dict observation space - get shape for each observation group
            actor_obs_shape = {}
            for k in env.observation_space.keys():
                if isinstance(env.observation_space[k], gym.spaces.Dict):
                    # Skip nested dicts (not supported)
                    continue
                actor_obs_shape[k] = env.observation_space[k].shape[1:]  # Remove batch dim
        else:
            # Tensor observation space
            actor_obs_shape = env.observation_space.shape[1:]  # Remove batch dim

        # Use learner's internal action dimension for storage (not env's)
        # This ensures storage uses the actual control dimension, not padded dimension
        self._learner_action_dim = self.learner.planner.sampling.nu

        self.learner.init_storage(
            num_envs=self.num_envs,
            num_steps_per_env=self._num_steps_per_env, # -1 since we store (s_t, a_t, s_t+1)
            actor_obs_shape=actor_obs_shape,
            critic_obs_shape=None, # No privileged observations
            action_shape=[self._learner_action_dim],
        )

        if getattr(self.cfg, "resume_from_checkpoint", None):
            self.load(self.cfg.resume_from_checkpoint, load_optimizer=False)
            print(f"[INFO] Warm-started planner from {self.cfg.resume_from_checkpoint}")
        if self.action_scale is not None:
            print(f"[INFO] Scaling learner actions before env.step: {self.action_scale.detach().cpu().tolist()}")
        if self.teacher_policy is not None:
            print(f"[INFO] Online teacher bootstrap for {self.teacher_bootstrap_steps} env steps")

        # Log
        if self.cfg.logger == "wandb":
            # from rsl_rl.utils.wandb_utils import WandbSummarylogger
            # self.logger = WandbSummarylogger(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            import wandb
            self.logger = wandb
        else:
            print("[INFO] No logger specified or not recognized.")
            self.logger = None

        if self.cfg.logger and self.cfg.vis_rollouts:
            assert self.learner.planner.vis is not None, "Policy must have visualization enabled for rollouts"
            self.rollouts_vid = RolloutsVideo(self.learner.planner.vis)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.render_during_training = bool(getattr(self.cfg, "render_during_training", False))
        self.render_interval_steps = max(1, int(getattr(self.cfg, "render_interval_steps", 1) or 1))
        if self.render_during_training:
            print(
                f"[INFO] Live training render enabled: interval_steps={self.render_interval_steps}",
                flush=True,
            )
        self.training_video_enabled = bool(getattr(self.log_cfg, "video_enabled", False))
        self.training_video_camera_name = "side_camera"
        self.training_video_length = int(getattr(self.log_cfg, "video_length", 0) or 0)
        self.training_video_resolution = tuple(getattr(self.log_cfg, "video_resolution", (640, 360)))
        self.training_video_crf = int(getattr(self.log_cfg, "video_crf", 30) or 30)
        self.training_video_fps = int(getattr(self.log_cfg, "video_fps", 10) or 10)
        self._training_video_frames: list[np.ndarray] = []
        if self.training_video_enabled:
            os.makedirs(os.path.join(self.log_dir, "videos"), exist_ok=True)
            scene = getattr(self.env.unwrapped, "scene", None)
            has_camera = scene is not None and self.training_video_camera_name in list(scene.keys())
            if has_camera:
                print(
                    "[INFO] Training video camera recorder enabled: "
                    f"camera={self.training_video_camera_name} "
                    f"interval_iters={self.video_interval_its} "
                    f"max_frames={self.training_video_length or self._num_steps_per_env}",
                    flush=True,
                )
            else:
                print(
                    f"[WARN] Training video requested, but camera {self.training_video_camera_name!r} "
                    "is not present in the scene. Videos will be skipped.",
                    flush=True,
                )
                self.training_video_enabled = False

    @staticmethod
    def _count_demo_transitions(demonstrations: torch.Tensor | Dict[str, torch.Tensor]) -> int:
        if isinstance(demonstrations, dict):
            first_value = next(iter(demonstrations.values()))
            return int(first_value.shape[0])
        return int(demonstrations.shape[0])

    def _termination_term_counts(self) -> dict[str, int]:
        termination_manager = getattr(self.env.unwrapped, "termination_manager", None)
        if termination_manager is None:
            return {}

        counts = {}
        for name in getattr(termination_manager, "active_terms", []):
            try:
                value = termination_manager.get_term(name)
            except Exception:
                continue
            if torch.is_tensor(value):
                counts[name] = int(value.detach().to(device="cpu").sum().item())
        return counts

    def _termination_term_mask(self, name: str) -> torch.Tensor | None:
        termination_manager = getattr(self.env.unwrapped, "termination_manager", None)
        if termination_manager is None:
            return None
        active_terms = getattr(termination_manager, "active_terms", [])
        if name not in active_terms:
            return None
        try:
            value = termination_manager.get_term(name)
        except Exception:
            return None
        if not torch.is_tensor(value):
            return torch.as_tensor(value, device=self.device).bool()
        return value.to(device=self.device).bool()

    def _finite_obs_mask(self, obs) -> torch.Tensor:
        if isinstance(obs, dict):
            mask = None
            for value in obs.values():
                current = torch.isfinite(value).flatten(start_dim=1).all(dim=1)
                mask = current if mask is None else (mask & current)
            if mask is None:
                return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return mask.to(device=self.device)
        return torch.isfinite(obs).flatten(start_dim=1).all(dim=1).to(device=self.device)

    def _omnireset_success_rate(self) -> float | None:
        reward_manager = getattr(self.env.unwrapped, "reward_manager", None)
        if reward_manager is None:
            return None
        try:
            progress = reward_manager.get_term_cfg("progress_context").func
        except Exception:
            return None

        success = getattr(progress, "success", None)
        if success is None:
            return None
        if torch.is_tensor(success):
            return float(success.detach().float().mean().to(device="cpu").item())
        return None

    def train_mode(self):
        self.learner.planner.train()

    def learn(self):

        # start learning
        obs, infos = self.env.reset()
        critic_obs = obs
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        failure_episode_start_obs = self._clone_policy_obs(obs)
        failure_episode_start_raw = self._clone_raw_policy_obs()
        failure_episode_min_xyz = self._obs_xyz_distance(obs)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.cfg.num_learning_iterations
        for it in tqdm(range(start_iter, tot_iter)):
            start = time.time()
            ep_stats = Stats()

            # Set current iteration on environment for curriculum learning
            self.env.unwrapped.current_iteration = it

            vis_ep = self.cfg.vis_rollouts and it % self.video_interval_its == 0
            iter_completed_rewards = []  # Track rewards from episodes completed this iteration
            iter_success_sum = 0.0
            iter_success_samples = 0
            iter_done_count = 0
            iter_terminated_count = 0
            iter_time_out_count = 0
            iter_term_counts: dict[str, int] = {}
            online_teacher_obs: list[torch.Tensor] = []
            online_teacher_actions: list[torch.Tensor] = []
            online_mppi_rank_obs: list[torch.Tensor] = []
            online_mppi_rank_teacher: list[torch.Tensor] = []
            online_mppi_rank_policy: list[torch.Tensor] = []
            online_mppi_rank_mppi: list[torch.Tensor] = []
            online_mppi_rank_blended: list[torch.Tensor] = []
            online_mppi_outcome_obs: list[torch.Tensor] = []
            online_mppi_outcome_policy: list[torch.Tensor] = []
            online_mppi_outcome_mppi: list[torch.Tensor] = []
            online_mppi_outcome_blended: list[torch.Tensor] = []
            online_mppi_outcome_good: list[torch.Tensor] = []
            online_mppi_outcome_bad: list[torch.Tensor] = []
            online_mppi_candidate_obs: list[torch.Tensor] = []
            online_mppi_candidate_policy: list[torch.Tensor] = []
            online_mppi_candidate_good: list[torch.Tensor] = []
            online_mppi_candidate_bad: list[torch.Tensor] = []
            online_mppi_candidate_good_mask: list[torch.Tensor] = []
            delayed_mppi_outcome_records: list[dict[str, torch.Tensor]] = []
            iter_min_xyz = self._obs_xyz_distance(obs)
            record_training_video = self._should_record_training_video(it)
            if record_training_video:
                self._training_video_frames = []
            # Rollout
            self.learner.eval()
            for i in range(self._num_steps_per_env):

                with torch.inference_mode():

                    # Sample actions from policy
                    teacher_actions = self._teacher_actions()
                    if self.policy_control_after_online_teacher_bc and it > start_iter:
                        teacher_actions = None
                    teacher_diag_actions = teacher_actions
                    if teacher_diag_actions is None and self.teacher_diag_always:
                        teacher_diag_actions = self._teacher_actions(ignore_alpha=True)

                    learner_actions = self.learner.act(obs, vis_rollouts=vis_ep) # TODO: critic obs
                    teacher_oracle_mppi_stats = {}
                    if teacher_diag_actions is not None and bool(getattr(self.cfg, "teacher_oracle_mppi_gate", False)):
                        mppi_tensors = getattr(self.learner, "_last_mppi_action_tensors", {})
                        policy_actions = mppi_tensors.get("policy")
                        mppi_actions = mppi_tensors.get("mppi")
                        blended_actions = mppi_tensors.get("blended")
                        if (
                            torch.is_tensor(policy_actions)
                            and torch.is_tensor(mppi_actions)
                            and torch.is_tensor(blended_actions)
                        ):
                            teacher_l2_policy = (policy_actions - teacher_diag_actions).norm(dim=-1)
                            teacher_l2_mppi = (mppi_actions - teacher_diag_actions).norm(dim=-1)
                            candidate_actions = getattr(self.learner.planner, "_prior_controls", None)
                            candidate_better_rate = None
                            candidate_best_l2 = None
                            candidate_best_idx = None
                            if torch.is_tensor(candidate_actions) and candidate_actions.ndim >= 4:
                                first_actions = candidate_actions.detach()[:, :, 0, :]
                                candidate_l2 = (first_actions - teacher_diag_actions.unsqueeze(1)).norm(dim=-1)
                                candidate_best_l2, candidate_best_idx = candidate_l2.min(dim=1)
                                candidate_worst_l2, candidate_worst_idx = candidate_l2.max(dim=1)
                                candidate_better_rate = (candidate_best_l2 < teacher_l2_policy).float().mean()
                                if getattr(self.learner.cfg, "online_mppi_candidate_rank_coeff", 0.0) > 0:
                                    best_gather = candidate_best_idx.view(-1, 1, 1).expand(-1, 1, first_actions.shape[-1])
                                    worst_gather = candidate_worst_idx.view(-1, 1, 1).expand(-1, 1, first_actions.shape[-1])
                                    online_mppi_candidate_obs.append(obs["policy"].detach().clone())
                                    online_mppi_candidate_policy.append(policy_actions.detach().clone())
                                    online_mppi_candidate_good.append(
                                        first_actions.gather(dim=1, index=best_gather).squeeze(1).detach().clone()
                                    )
                                    online_mppi_candidate_bad.append(
                                        first_actions.gather(dim=1, index=worst_gather).squeeze(1).detach().clone()
                                    )
                                    online_mppi_candidate_good_mask.append(
                                        (candidate_best_l2 < teacher_l2_policy).detach().clone()
                                    )
                                if bool(getattr(self.cfg, "teacher_oracle_mppi_pool_select", False)):
                                    gather_idx = candidate_best_idx.view(-1, 1, 1).expand(-1, 1, first_actions.shape[-1])
                                    mppi_actions = first_actions.gather(dim=1, index=gather_idx).squeeze(1)
                                    teacher_l2_mppi = candidate_best_l2
                                    blend = float(getattr(self.learner.cfg, "mppi_policy_blend", 1.0))
                                    delta = mppi_actions - policy_actions
                                    delta_l2_max = getattr(self.learner.cfg, "mppi_policy_delta_l2_max", None)
                                    if delta_l2_max is not None:
                                        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                                        delta = delta * (float(delta_l2_max) / delta_norm).clamp(max=1.0)
                                    blended_actions = (policy_actions + blend * delta).clamp(-1.0, 1.0)
                            margin = float(getattr(self.cfg, "teacher_oracle_mppi_margin", 0.0) or 0.0)
                            accept = teacher_l2_mppi + margin < teacher_l2_policy
                            learner_actions = torch.where(accept.unsqueeze(-1), blended_actions, policy_actions)
                            self.learner.transition.actions = learner_actions
                            mppi_tensors["executed"] = learner_actions.detach()
                            mppi_tensors["teacher_oracle_accept"] = accept.detach()
                            teacher_oracle_mppi_stats = {
                                "TeacherOracleMPPI/accept_rate": accept.float().mean().item(),
                                "TeacherOracleMPPI/policy_l2": teacher_l2_policy.mean().item(),
                                "TeacherOracleMPPI/mppi_l2": teacher_l2_mppi.mean().item(),
                                "TeacherOracleMPPI/mppi_better_rate": (
                                    teacher_l2_mppi < teacher_l2_policy
                                ).float().mean().item(),
                                "TeacherOracleMPPI/executed_policy_l2": (
                                    learner_actions - teacher_diag_actions
                                ).norm(dim=-1).mean().item(),
                            }
                            if candidate_better_rate is not None:
                                teacher_oracle_mppi_stats.update({
                                    "TeacherOracleMPPI/candidate_pool_better_rate": candidate_better_rate.item(),
                                    "TeacherOracleMPPI/candidate_pool_best_l2": candidate_best_l2.mean().item(),
                                    "TeacherOracleMPPI/candidate_pool_best_idx": candidate_best_idx.float().mean().item(),
                                })
                    intervention_mask = None
                    if (
                        teacher_actions is None
                        and teacher_diag_actions is not None
                        and self.teacher_intervention_cos_threshold is not None
                    ):
                        learner_teacher_cos = torch.nn.functional.cosine_similarity(
                            teacher_diag_actions, learner_actions, dim=-1
                        )
                        intervention_mask = learner_teacher_cos < float(self.teacher_intervention_cos_threshold)

                    residual_actions = None
                    if (
                        teacher_actions is None
                        and teacher_diag_actions is not None
                        and self._teacher_residual_beta(it, start_iter, tot_iter) > 0.0
                    ):
                        beta = self._teacher_residual_beta(it, start_iter, tot_iter)
                        residual_actions = (learner_actions + beta * (teacher_diag_actions - learner_actions)).clamp(
                            -1.0, 1.0
                        )

                    if teacher_actions is None and intervention_mask is None:
                        actions = residual_actions if residual_actions is not None else learner_actions
                    elif teacher_actions is None:
                        fallback_actions = residual_actions if residual_actions is not None else learner_actions
                        actions = torch.where(intervention_mask.unsqueeze(-1), teacher_diag_actions, fallback_actions)
                    else:
                        alpha = self._teacher_alpha()
                        if alpha < 1.0:
                            actions = (alpha * teacher_actions + (1.0 - alpha) * learner_actions).clamp(-1.0, 1.0)
                        else:
                            actions = teacher_actions
                    self.learner.transition.actions = actions
                    self.learner.transition.observations = obs
                    self.learner.transition.critic_observations = obs

                    # Step environment
                    env_actions = self._scale_actions(actions).to(self.env.unwrapped.device)
                    raw_obs_before_step = self._clone_raw_policy_obs()
                    next_obs, rewards, terms, truncs, infos = self.env.step(env_actions)
                    if self.render_during_training and i % self.render_interval_steps == 0:
                        try:
                            self.env.unwrapped.sim.render()
                        except Exception as exc:
                            if i == 0:
                                print(f"[WARN] Live training render failed: {exc}", flush=True)
                    if record_training_video and self._should_capture_training_video_frame(i):
                        self._capture_training_video_frame()
                    self.tot_timesteps += self.num_envs
                    action_stats = self._action_stats(actions, env_actions, infos)
                    action_stats["Control/teacher_executing"] = float(teacher_actions is not None)
                    action_stats["Control/policy_control_after_online_teacher_bc"] = float(
                        self.policy_control_after_online_teacher_bc
                    )
                    action_stats["Control/teacher_dagger_collect"] = float(self.teacher_dagger_collect)
                    action_stats["Control/intervention_rate"] = (
                        0.0 if intervention_mask is None else intervention_mask.float().mean().item()
                    )
                    action_stats["Control/teacher_residual_beta"] = self._teacher_residual_beta(
                        it, start_iter, tot_iter
                    )
                    action_stats["Control/residual_active"] = float(residual_actions is not None)
                    action_stats.update(getattr(self.learner, "_last_mppi_action_stats", {}))
                    action_stats.update(teacher_oracle_mppi_stats)
                    current_xyz = self._obs_xyz_distance(obs)
                    next_xyz = self._obs_xyz_distance(next_obs)
                    if next_xyz is not None:
                        iter_min_xyz = next_xyz if iter_min_xyz is None else torch.minimum(iter_min_xyz, next_xyz)
                        if failure_episode_min_xyz is not None:
                            failure_episode_min_xyz = torch.minimum(failure_episode_min_xyz, next_xyz)
                    action_stats.update(self._dynamics_one_step_diag(obs, actions, next_obs))
                    outcome_horizon = int(getattr(self.learner.cfg, "online_mppi_outcome_horizon", 1))
                    if (
                        current_xyz is not None
                        and next_xyz is not None
                        and getattr(self.learner.cfg, "online_mppi_outcome_rank_coeff", 0.0) > 0
                    ):
                        mppi_tensors = getattr(self.learner, "_last_mppi_action_tensors", {})
                        mppi_actions = mppi_tensors.get("mppi")
                        policy_actions_for_outcome = mppi_tensors.get("policy")
                        blended_actions = mppi_tensors.get("blended")
                        if (
                            torch.is_tensor(mppi_actions)
                            and torch.is_tensor(policy_actions_for_outcome)
                            and torch.is_tensor(blended_actions)
                        ):
                            outcome_abnormal = self._termination_term_mask("abnormal_robot")
                            if outcome_abnormal is None:
                                outcome_abnormal = torch.zeros_like(next_xyz, dtype=torch.bool, device=self.device)
                            else:
                                outcome_abnormal = outcome_abnormal.to(device=self.device, dtype=torch.bool)
                            if outcome_horizon <= 1:
                                good_margin = float(getattr(self.learner.cfg, "online_mppi_outcome_good_margin", 0.0))
                                bad_margin = float(getattr(self.learner.cfg, "online_mppi_outcome_bad_margin", 0.0))
                                improvement = current_xyz.detach() - next_xyz.detach()
                                good_mask = (improvement > good_margin) & (~outcome_abnormal)
                                bad_mask = (improvement < -bad_margin) | outcome_abnormal
                                online_mppi_outcome_obs.append(obs["policy"].detach().clone())
                                online_mppi_outcome_policy.append(policy_actions_for_outcome.detach().clone())
                                online_mppi_outcome_mppi.append(mppi_actions.detach().clone())
                                online_mppi_outcome_blended.append(blended_actions.detach().clone())
                                online_mppi_outcome_good.append(good_mask.detach().clone())
                                online_mppi_outcome_bad.append(bad_mask.detach().clone())
                                action_stats["MPPIOutcome/good_rate"] = good_mask.float().mean().item()
                                action_stats["MPPIOutcome/bad_rate"] = bad_mask.float().mean().item()
                                action_stats["MPPIOutcome/improvement_mean"] = improvement.float().mean().item()
                            else:
                                delayed_mppi_outcome_records.append({
                                    "obs": obs["policy"].detach().clone(),
                                    "policy": policy_actions_for_outcome.detach().clone(),
                                    "mppi": mppi_actions.detach().clone(),
                                    "blended": blended_actions.detach().clone(),
                                    "current_xyz": current_xyz.detach().clone(),
                                    "next_xyz": next_xyz.detach().clone(),
                                    "abnormal": outcome_abnormal.detach().clone(),
                                })
                    if teacher_diag_actions is not None and learner_actions is not None:
                        policy_mean_actions = self.learner._bc_policy_mean_action(obs)
                        if (
                            (
                                getattr(self.learner.cfg, "online_teacher_bc_steps", 0) > 0
                                or getattr(self.learner.cfg, "residual_adapter_steps", 0) > 0
                            )
                            and (teacher_actions is not None or self.teacher_dagger_collect)
                        ):
                            teacher_obs_key = "policy"
                            if (
                                getattr(self.learner.cfg, "residual_adapter_use_raw_obs", False)
                                and getattr(self.learner.cfg, "online_teacher_bc_steps", 0) <= 0
                            ):
                                teacher_obs_key = getattr(self.learner.cfg, "raw_base_policy_obs_key", "raw_policy")
                            online_teacher_obs.append(obs[teacher_obs_key].detach().clone())
                            label_actions = actions if self.teacher_dagger_label_mode == "executed" else teacher_diag_actions
                            online_teacher_actions.append(label_actions.detach().clone())
                        if getattr(self.learner.cfg, "online_mppi_reward_rank_coeff", 0.0) > 0:
                            mppi_tensors = getattr(self.learner, "_last_mppi_action_tensors", {})
                            mppi_actions = mppi_tensors.get("mppi")
                            buffered_policy_actions = mppi_tensors.get("policy", policy_mean_actions)
                            blended_actions = mppi_tensors.get("blended")
                            if torch.is_tensor(mppi_actions):
                                online_mppi_rank_obs.append(obs["policy"].detach().clone())
                                online_mppi_rank_teacher.append(teacher_diag_actions.detach().clone())
                                online_mppi_rank_policy.append(buffered_policy_actions.detach().clone())
                                online_mppi_rank_mppi.append(mppi_actions.detach().clone())
                                if torch.is_tensor(blended_actions):
                                    online_mppi_rank_blended.append(blended_actions.detach().clone())
                        action_stats.update(
                            self._teacher_action_diag(
                                teacher_diag_actions, learner_actions, actions, policy_mean_actions
                            )
                        )
                        action_stats.update(
                            self._planner_action_ranking_diag(
                                obs,
                                teacher_diag_actions,
                                learner_actions,
                                policy_mean_actions,
                                getattr(self.learner, "_last_mppi_action_tensors", {}),
                            )
                        )
                        action_stats["Control/dagger_label_executed"] = float(
                            self.teacher_dagger_label_mode == "executed"
                        )

                    # Move to the agent device
                    dones = (terms | truncs).to(dtype=torch.long)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    iter_done_count += int(dones.detach().to(device="cpu").sum().item())
                    iter_terminated_count += int(terms.detach().to(device="cpu").sum().item())
                    iter_time_out_count += int(truncs.detach().to(device="cpu").sum().item())
                    for term_name, term_count in self._termination_term_counts().items():
                        iter_term_counts[term_name] = iter_term_counts.get(term_name, 0) + term_count
                    success_rate = self._omnireset_success_rate()
                    if success_rate is not None:
                        iter_success_sum += success_rate
                        iter_success_samples += 1

                    # Process env step and store in buffer. Keep abnormal resets active
                    # for simulator safety, but prevent those transitions from entering
                    # replay sampling for dynamics/reward/value learning.
                    valid_transitions = torch.ones_like(dones, dtype=torch.bool, device=self.device)
                    abnormal_mask = self._termination_term_mask("abnormal_robot")
                    if abnormal_mask is not None:
                        valid_transitions &= ~abnormal_mask
                    valid_transitions &= self._finite_obs_mask(obs)
                    valid_transitions &= self._finite_obs_mask(next_obs)
                    valid_transitions &= torch.isfinite(actions).flatten(start_dim=1).all(dim=1)
                    valid_transitions &= torch.isfinite(rewards)
                    stats = self.learner.process_env_step(
                        rewards, dones, infos, next_obs, valid_transitions=valid_transitions
                    )

                    if self.log_dir is not None:

                        # Log information
                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        # Update step stats
                        ep_stats.update(stats)
                        ep_stats.update(action_stats)
                        if infos is not None and 'log' in infos:
                            ep_stats.update(infos['log'])

                        # Clear data for completed episodes
                        # Use 1D indices and guard empty to avoid CUDA device-side asserts
                        new_ids = (dones > 0).nonzero(as_tuple=True)[0]
                        if new_ids.numel() > 0:
                            self._write_failure_strat_records(
                                it=it,
                                step=i,
                                env_ids=new_ids,
                                episode_start_obs=failure_episode_start_obs,
                                episode_start_raw=failure_episode_start_raw,
                                episode_min_xyz=failure_episode_min_xyz,
                                final_obs=obs,
                                final_raw=raw_obs_before_step,
                                rewards=cur_reward_sum,
                                lengths=cur_episode_length,
                                terms=terms,
                                truncs=truncs,
                                teacher_actions=teacher_diag_actions,
                                policy_actions=learner_actions,
                                executed_actions=actions,
                            )
                            # cur_reward_sum and cur_episode_length are 1D (num_envs,)
                            completed_rewards = cur_reward_sum[new_ids].cpu().numpy().tolist()
                            iter_completed_rewards.extend(completed_rewards)
                            rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0
                            if failure_episode_start_obs is not None:
                                next_policy = self._clone_policy_obs(next_obs)
                                if next_policy is not None:
                                    failure_episode_start_obs[new_ids] = next_policy[new_ids]
                            if failure_episode_start_raw is not None:
                                next_raw = self._clone_raw_policy_obs()
                                if next_raw is not None:
                                    failure_episode_start_raw[new_ids] = next_raw[new_ids]
                            if failure_episode_min_xyz is not None and next_xyz is not None:
                                failure_episode_min_xyz[new_ids] = next_xyz[new_ids]

                if vis_ep:
                    self.rollouts_vid.update_video()

                # Update obs
                obs = next_obs

            # End of rollout collection
            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop

            if delayed_mppi_outcome_records and getattr(self.learner.cfg, "online_mppi_outcome_rank_coeff", 0.0) > 0:
                outcome_horizon = max(1, int(getattr(self.learner.cfg, "online_mppi_outcome_horizon", 1)))
                next_xyz_seq = torch.stack([r["next_xyz"] for r in delayed_mppi_outcome_records], dim=0)
                abnormal_seq = torch.stack([r["abnormal"] for r in delayed_mppi_outcome_records], dim=0)
                good_margin = float(getattr(self.learner.cfg, "online_mppi_outcome_good_margin", 0.0))
                bad_margin = float(getattr(self.learner.cfg, "online_mppi_outcome_bad_margin", 0.0))
                delayed_good = []
                delayed_bad = []
                delayed_improvement = []
                for rec_idx, record in enumerate(delayed_mppi_outcome_records):
                    end = min(len(delayed_mppi_outcome_records), rec_idx + outcome_horizon)
                    future_xyz = next_xyz_seq[rec_idx:end]
                    future_abnormal = abnormal_seq[rec_idx:end].any(dim=0)
                    best_future_xyz = future_xyz.min(dim=0).values
                    improvement = record["current_xyz"] - best_future_xyz
                    delayed_improvement.append(improvement)
                    delayed_good.append((improvement > good_margin) & (~future_abnormal))
                    delayed_bad.append((improvement < -bad_margin) | future_abnormal)
                online_mppi_outcome_obs.append(torch.cat([r["obs"] for r in delayed_mppi_outcome_records], dim=0))
                online_mppi_outcome_policy.append(torch.cat([r["policy"] for r in delayed_mppi_outcome_records], dim=0))
                online_mppi_outcome_mppi.append(torch.cat([r["mppi"] for r in delayed_mppi_outcome_records], dim=0))
                online_mppi_outcome_blended.append(torch.cat([r["blended"] for r in delayed_mppi_outcome_records], dim=0))
                online_mppi_outcome_good.append(torch.stack(delayed_good, dim=0).flatten())
                online_mppi_outcome_bad.append(torch.stack(delayed_bad, dim=0).flatten())
                self._last_delayed_mppi_outcome_stats = {
                    "MPPIOutcome/delayed_good_rate": online_mppi_outcome_good[-1].float().mean().item(),
                    "MPPIOutcome/delayed_bad_rate": online_mppi_outcome_bad[-1].float().mean().item(),
                    "MPPIOutcome/delayed_improvement_mean": torch.stack(delayed_improvement, dim=0).float().mean().item(),
                    "MPPIOutcome/horizon": float(outcome_horizon),
                }
            else:
                self._last_delayed_mppi_outcome_stats = {}

            # Log returns
            ep_ret = self.learner.storage.rewards.sum(0).cpu().numpy()
            max_return, mean_return, min_return = ep_ret.max(), ep_ret.mean(), ep_ret.min()

            # Update learner
            # Note: we keep arguments here since locals() loads them
            train_stats = {}
            self.learner.train()
            teacher_bc_steps = int(getattr(self.learner.cfg, "online_teacher_bc_steps", 0))
            residual_adapter_steps = int(getattr(self.learner.cfg, "residual_adapter_steps", 0))
            if online_teacher_obs and (teacher_bc_steps > 0 or residual_adapter_steps > 0):
                online_obs = torch.cat(online_teacher_obs, dim=0)
                online_actions = torch.cat(online_teacher_actions, dim=0)
                online_obs, online_actions = self._update_dagger_buffer(online_obs, online_actions)
                if teacher_bc_steps > 0:
                    train_stats.update(
                        self.learner.online_teacher_bc_update(
                            {"policy": online_obs},
                            online_actions,
                        )
                    )
                if residual_adapter_steps > 0:
                    train_stats.update(
                        self.learner.residual_adapter_update(
                            {"policy": online_obs},
                            online_actions,
                        )
                    )
                train_stats["OnlineTeacherBC/buffer_size"] = float(online_obs.shape[0])
            if online_mppi_rank_obs and getattr(self.learner.cfg, "online_mppi_reward_rank_coeff", 0.0) > 0:
                online_obs = torch.cat(online_mppi_rank_obs, dim=0)
                online_teacher = torch.cat(online_mppi_rank_teacher, dim=0)
                online_policy = torch.cat(online_mppi_rank_policy, dim=0)
                online_mppi = torch.cat(online_mppi_rank_mppi, dim=0)
                online_blended = (
                    torch.cat(online_mppi_rank_blended, dim=0)
                    if online_mppi_rank_blended
                    else None
                )
                train_stats.update(
                    self.learner.online_mppi_reward_rank_update(
                        online_obs,
                        online_teacher,
                        online_policy,
                        online_mppi,
                        online_blended,
                    )
                )
            if online_mppi_outcome_obs and getattr(self.learner.cfg, "online_mppi_outcome_rank_coeff", 0.0) > 0:
                train_stats.update(
                    self.learner.online_mppi_outcome_rank_update(
                        torch.cat(online_mppi_outcome_obs, dim=0),
                        torch.cat(online_mppi_outcome_policy, dim=0),
                        torch.cat(online_mppi_outcome_mppi, dim=0),
                        torch.cat(online_mppi_outcome_blended, dim=0),
                        torch.cat(online_mppi_outcome_good, dim=0),
                        torch.cat(online_mppi_outcome_bad, dim=0),
                    )
                )
                train_stats.update(getattr(self, "_last_delayed_mppi_outcome_stats", {}))
            if online_mppi_candidate_obs and getattr(self.learner.cfg, "online_mppi_candidate_rank_coeff", 0.0) > 0:
                train_stats.update(
                    self.learner.online_mppi_candidate_rank_update(
                        torch.cat(online_mppi_candidate_obs, dim=0),
                        torch.cat(online_mppi_candidate_policy, dim=0),
                        torch.cat(online_mppi_candidate_good, dim=0),
                        torch.cat(online_mppi_candidate_bad, dim=0),
                        torch.cat(online_mppi_candidate_good_mask, dim=0),
                    )
                )
            train_stats.update(self.learner.update(iteration=it)) # Can return metrics for loss logging

            self.learner.storage.clear()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # TODO
            # Logging info and save checkpoint

            fps = int(self._num_steps_per_env * self.num_envs / (collection_time + learn_time))
            stats = {
                "Perf/collection_time": collection_time,
                "Perf/learn_time": learn_time,
                "Perf/fps": fps,
                "Env/mean_reward": 0 if len(rewbuffer) == 0 else statistics.mean(rewbuffer),
                "Env/mean_length": 0 if len(lenbuffer) == 0 else statistics.mean(lenbuffer),
                "Env/ep_mean_return": 0 if len(iter_completed_rewards) == 0 else statistics.mean(iter_completed_rewards),
                "Env/max_return": max_return,
                "Env/mean_return": mean_return,
                "Env/min_return": min_return,
                "Env/done_count": iter_done_count,
                "Env/terminated_count": iter_terminated_count,
                "Env/time_out_count": iter_term_counts.get("time_out", iter_time_out_count),
                "Env/abnormal_robot_count": iter_term_counts.get("abnormal_robot", 0),
                "Data/demo_transitions": self.demo_transition_count,
                "it": it,
                "tot_iter": tot_iter,
            }
            if iter_success_samples > 0:
                stats["Env/success_rate"] = iter_success_sum / iter_success_samples
            if iter_min_xyz is not None:
                stats["Env/min_xyz_mean"] = iter_min_xyz.detach().float().mean().to(device="cpu").item()

            train_stats.update(stats)
            train_stats.update(ep_stats.mean())
            video_path = None
            if record_training_video:
                video_path = self._save_training_video(it)
                if video_path is not None:
                    train_stats["Video/training_frames"] = len(self._training_video_frames)

            if self.log_dir is not None and self.logger:

                self.logger.log(train_stats)
                if video_path is not None and not self.log_cfg.no_wandb:
                    wandb.log(
                        {
                            "Training Video": wandb.Video(
                                video_path,
                                fps=self.training_video_fps,
                                format="mp4",
                            )
                        },
                        commit=False,
                    )

                if hasattr(self.cfg, "log_task_stat"):
                    # Log task-specific statistics
                    task_stat = task_stats(ep_stats.stats, self.cfg.log_task_stat)
                    train_stats.update(task_stat)

            if self.log_dir is not None:
                metrics_path = os.path.join(self.log_dir, "metrics.jsonl")
                with open(metrics_path, "a", encoding="utf-8") as f:
                    json.dump(
                        {
                            key: (value.item() if hasattr(value, "item") else value)
                            for key, value in train_stats.items()
                            if isinstance(value, (int, float, bool)) or hasattr(value, "item")
                        },
                        f,
                        sort_keys=True,
                    )
                    f.write("\n")

                # Log Rollout visualization
                if vis_ep:
                    vid_save_dir = os.path.join(self.log_dir, "rollouts_vis")
                    path_to_vid = self.rollouts_vid.save_video(
                        output_dir=vid_save_dir,
                        episode_num=it,
                        frame_rate=10
                    )
                    if not self.log_cfg.no_wandb:
                        wandb.log({"Rollouts Video": wandb.Video(path_to_vid)}, commit=False)
                        self.rollouts_vid.reset()

            # Save model checkpoint (always, regardless of logger)
            if self.log_dir is not None and self.log_cfg.checkpoint_every and it % self.log_cfg.checkpoint_every == 0:
                self.save(postfix=f"{it}")

            # Print progress to stdout every iteration
            print(
                f"[iter {it:4d}/{tot_iter}] "
                f"fps={train_stats.get('Perf/fps', 0):4.0f}  "
                f"rew={train_stats.get('Env/mean_reward', 0):.4f}  "
                f"ep_ret={train_stats.get('Env/ep_mean_return', 0):.3f}  "
                f"succ={train_stats.get('Env/success_rate', float('nan')):.3f}  "
                f"done={train_stats.get('Env/done_count', 0)}  "
                f"abn={train_stats.get('Env/abnormal_robot_count', 0)}  "
                f"to={train_stats.get('Env/time_out_count', 0)}  "
                f"disc_demo={train_stats.get('Reward/mean_demo_reward', float('nan')):.3f}  "
                f"disc_gen={train_stats.get('Reward/mean_gen_reward', float('nan')):.3f}  "
                f"dyn_loss={train_stats.get('Dyn/mean_loss', float('nan')):.4f}",
                flush=True,
            )

        # Save the final model after training
        if self.log_dir is not None:
            self.save(postfix=f"{tot_iter}")
        self._export_dagger_buffer()

    def save(self, postfix: str=""):

        saved_dict = {
            "model_state_dict": self.learner.planner.state_dict(),
            "reward_optimizer_state_dict": self.learner._reward_opt.state_dict(),
            "value_optimizer_state_dict": self.learner._value_opt.state_dict(),
            "iter": self.current_learning_iteration,
        }
        if self.action_scale is not None:
            saved_dict["action_scale"] = self.action_scale.detach().cpu()
        if getattr(self.learner, "_residual_adapter", None) is not None:
            saved_dict["residual_adapter_state_dict"] = self.learner._residual_adapter.state_dict()
        if getattr(self.learner, "_residual_adapter_opt", None) is not None:
            saved_dict["residual_adapter_optimizer_state_dict"] = self.learner._residual_adapter_opt.state_dict()
        if getattr(self.learner, "_candidate_selector", None) is not None:
            saved_dict["candidate_selector_state_dict"] = self.learner._candidate_selector.state_dict()
        if getattr(self.learner, "_candidate_selector_opt", None) is not None:
            saved_dict["candidate_selector_optimizer_state_dict"] = self.learner._candidate_selector_opt.state_dict()

        model_path = os.path.join(self.log_dir, "models", f"model_{postfix}.pt")
        torch.save(saved_dict, model_path)

        # Save to external logger
        if self.logger:
            self.logger.save(model_path, base_path = os.path.dirname(model_path))


    def load(self, path: str, load_optimizer=True):
        ''' Load model from path '''

        saved_dict = torch.load(path, map_location=self.device, weights_only=False)

        model_state = saved_dict["model_state_dict"]
        if getattr(self.cfg, "resume_partial_ignore_mismatch", False):
            current_state = self.learner.planner.state_dict()
            compatible_state = {
                key: value
                for key, value in model_state.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            skipped = sorted(set(model_state) - set(compatible_state))
            current_state.update(compatible_state)
            self.learner.planner.load_state_dict(current_state)
            print(
                "[INFO] Partial warm-start: "
                f"loaded={len(compatible_state)} skipped={len(skipped)}",
                flush=True,
            )
            if skipped:
                print(
                    "[INFO] Partial warm-start skipped keys: "
                    + ", ".join(skipped[:12])
                    + (" ..." if len(skipped) > 12 else ""),
                    flush=True,
                )
        else:
            self.learner.planner.load_state_dict(model_state)
        if (
            getattr(self.learner, "_residual_adapter", None) is not None
            and "residual_adapter_state_dict" in saved_dict
        ):
            self.learner._residual_adapter.load_state_dict(saved_dict["residual_adapter_state_dict"])
        if (
            getattr(self.learner, "_candidate_selector", None) is not None
            and "candidate_selector_state_dict" in saved_dict
        ):
            self.learner._candidate_selector.load_state_dict(saved_dict["candidate_selector_state_dict"])
        if "action_scale" in saved_dict:
            self.action_scale = self._make_action_scale(saved_dict["action_scale"])
        if load_optimizer and "reward_optimizer_state_dict" in saved_dict: # Does this point to the correct params?
            self.learner._reward_opt.load_state_dict(saved_dict["reward_optimizer_state_dict"])
        if load_optimizer and "value_optimizer_state_dict" in saved_dict:
            self.learner._value_opt.load_state_dict(saved_dict["value_optimizer_state_dict"])
        if (
            load_optimizer
            and getattr(self.learner, "_residual_adapter_opt", None) is not None
            and "residual_adapter_optimizer_state_dict" in saved_dict
        ):
            self.learner._residual_adapter_opt.load_state_dict(saved_dict["residual_adapter_optimizer_state_dict"])
        if (
            load_optimizer
            and getattr(self.learner, "_candidate_selector_opt", None) is not None
            and "candidate_selector_optimizer_state_dict" in saved_dict
        ):
            self.learner._candidate_selector_opt.load_state_dict(saved_dict["candidate_selector_optimizer_state_dict"])

    @property
    def video_interval_its(self):
        return max(1, self.log_cfg.video_interval // self._num_steps_per_env)

    def _should_record_training_video(self, iteration: int) -> bool:
        return bool(self.training_video_enabled and iteration % self.video_interval_its == 0)

    def _should_capture_training_video_frame(self, step: int) -> bool:
        return self.training_video_length <= 0 or len(self._training_video_frames) < self.training_video_length

    def _capture_training_video_frame(self) -> None:
        try:
            camera = self.env.unwrapped.scene[self.training_video_camera_name]
            rgb = camera.data.output.get("rgb")
            if rgb is None:
                return
            frame = rgb[0].detach().to(device="cpu").numpy()
            if frame.dtype != np.uint8:
                if frame.size and frame.max() <= 1.0:
                    frame = frame * 255.0
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            elif frame.shape[-1] > 3:
                frame = frame[:, :, :3]
            if frame.size:
                self._training_video_frames.append(frame)
        except Exception as exc:
            if len(self._training_video_frames) == 0:
                print(f"[WARN] Training video frame capture failed: {exc}", flush=True)

    def _save_training_video(self, iteration: int) -> str | None:
        if not self._training_video_frames:
            print(f"[WARN] No training video frames captured for iter {iteration}; skipping video.", flush=True)
            return None
        video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        path = os.path.join(video_dir, f"training_iter_{iteration:04d}.mp4")
        imageio.mimsave(
            path,
            self._training_video_frames,
            fps=self.training_video_fps,
            macro_block_size=1,
            quality=None,
            output_params=["-crf", str(self.training_video_crf), "-preset", "veryfast"],
        )
        print(
            f"[INFO] Training video saved: {path} ({len(self._training_video_frames)} frames)",
            flush=True,
        )
        return path

    def _make_action_scale(self, value):
        if value is None:
            return None
        scale = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if scale.numel() != self.learner.planner.sampling.nu:
            raise ValueError(
                f"action_scale has {scale.numel()} dims, expected {self.learner.planner.sampling.nu}"
            )
        return scale.view(1, -1)

    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_scale is None:
            return actions
        return actions * self.action_scale.to(device=actions.device, dtype=actions.dtype)

    def _update_dagger_buffer(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.teacher_dagger_buffer_size <= 0:
            return obs, actions

        obs = obs.detach()
        actions = actions.detach()
        if self._dagger_obs_buffer is None:
            self._dagger_obs_buffer = obs
            self._dagger_action_buffer = actions
        else:
            self._dagger_obs_buffer = torch.cat((self._dagger_obs_buffer, obs), dim=0)
            self._dagger_action_buffer = torch.cat((self._dagger_action_buffer, actions), dim=0)

        if self._dagger_obs_buffer.shape[0] > self.teacher_dagger_buffer_size:
            keep = self.teacher_dagger_buffer_size
            self._dagger_obs_buffer = self._dagger_obs_buffer[-keep:]
            self._dagger_action_buffer = self._dagger_action_buffer[-keep:]

        return self._dagger_obs_buffer, self._dagger_action_buffer

    def _export_dagger_buffer(self) -> None:
        if not self.teacher_dagger_export_path:
            return
        if self._dagger_obs_buffer is None or self._dagger_action_buffer is None:
            print(f"[WARN] Requested DAgger export but buffer is empty: {self.teacher_dagger_export_path}", flush=True)
            return

        obs = self._dagger_obs_buffer.detach().cpu().float()
        actions = self._dagger_action_buffer.detach().cpu().float()
        raw_limit = getattr(self.learner.cfg, "demo_action_raw_limit", None)
        if raw_limit is not None:
            scale = torch.as_tensor(raw_limit, dtype=torch.float32).view(1, -1)
            raw_actions = actions * scale
        elif self.action_scale is not None:
            scale = self.action_scale.detach().cpu().float().view(1, -1)
            raw_actions = actions * scale
        else:
            raw_actions = actions
        export = {
            "data": {
                "obs": {"policy": obs},
                "next_obs": {"policy": obs.clone()},
                "actions": raw_actions,
            },
            "meta": {
                "source": "teacher_dagger_buffer",
                "actions": "raw_env_units",
                "num_samples": int(obs.shape[0]),
                "label_mode": self.teacher_dagger_label_mode,
            },
        }
        path = self.teacher_dagger_export_path
        if not os.path.isabs(path) and self.log_dir is not None:
            path = os.path.join(self.log_dir, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(export, path)
        print(f"[INFO] Exported DAgger BC dataset: {path} samples={obs.shape[0]}", flush=True)

    def _teacher_residual_beta(self, it: int, start_iter: int, tot_iter: int) -> float:
        if self.teacher_residual_final_beta is None:
            return self.teacher_residual_beta
        denom = max(1, (tot_iter - start_iter) - 1)
        frac = min(1.0, max(0.0, float(it - start_iter) / float(denom)))
        return self.teacher_residual_beta + frac * (
            self.teacher_residual_final_beta - self.teacher_residual_beta
        )

    def _teacher_alpha(self) -> float:
        if self.tot_timesteps < self.teacher_bootstrap_steps:
            return 1.0
        if self.teacher_blend_steps <= 0:
            return 0.0
        blend_elapsed = self.tot_timesteps - self.teacher_bootstrap_steps
        return max(0.0, 1.0 - float(blend_elapsed) / float(self.teacher_blend_steps))

    def _policy_obs_tensor(self, obs) -> torch.Tensor | None:
        policy_obs = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
        return policy_obs if torch.is_tensor(policy_obs) else None

    def _clone_policy_obs(self, obs) -> torch.Tensor | None:
        policy_obs = self._policy_obs_tensor(obs)
        if policy_obs is None:
            return None
        return policy_obs.detach().clone()

    def _clone_raw_policy_obs(self) -> torch.Tensor | None:
        raw_obs = getattr(self.env, "last_raw_policy_obs", None)
        if not torch.is_tensor(raw_obs):
            return None
        return raw_obs.detach().clone()

    def _peg_in_hole_xyz(self, obs_tensor: torch.Tensor | None) -> torch.Tensor | None:
        if not torch.is_tensor(obs_tensor):
            return None
        dim = obs_tensor.shape[-1]
        if dim == 215:
            return obs_tensor[..., 24:27]
        if dim >= 3:
            return obs_tensor[..., 0:3]
        return None

    def _obs_xyz_distance(self, obs) -> torch.Tensor | None:
        policy_obs = self._policy_obs_tensor(obs)
        peg_in_hole = self._peg_in_hole_xyz(policy_obs)
        if peg_in_hole is None:
            return None
        return peg_in_hole.norm(dim=-1).to(device=self.device)

    def _omnireset_success_mask(self) -> torch.Tensor | None:
        reward_manager = getattr(self.env.unwrapped, "reward_manager", None)
        if reward_manager is None:
            return None
        try:
            progress = reward_manager.get_term_cfg("progress_context").func
        except Exception:
            return None
        success = getattr(progress, "success", None)
        if not torch.is_tensor(success):
            return None
        return success.to(device=self.device, dtype=torch.bool)

    @staticmethod
    def _record_vec(tensor: torch.Tensor | None, env_id: int, start: int, end: int) -> list[float] | None:
        if not torch.is_tensor(tensor) or tensor.shape[-1] < end:
            return None
        return tensor[env_id, start:end].detach().float().to(device="cpu").tolist()

    @staticmethod
    def _record_scalar(tensor: torch.Tensor | None, env_id: int) -> float | None:
        if not torch.is_tensor(tensor):
            return None
        return float(tensor[env_id].detach().float().to(device="cpu").item())

    def _write_failure_strat_records(
        self,
        *,
        it: int,
        step: int,
        env_ids: torch.Tensor,
        episode_start_obs: torch.Tensor | None,
        episode_start_raw: torch.Tensor | None,
        episode_min_xyz: torch.Tensor | None,
        final_obs,
        final_raw: torch.Tensor | None,
        rewards: torch.Tensor,
        lengths: torch.Tensor,
        terms: torch.Tensor,
        truncs: torch.Tensor,
        teacher_actions: torch.Tensor | None,
        policy_actions: torch.Tensor | None,
        executed_actions: torch.Tensor | None,
    ) -> None:
        if not self.failure_strat_export_path:
            return
        final_policy = self._policy_obs_tensor(final_obs)
        success_mask = self._omnireset_success_mask()
        abnormal_mask = self._termination_term_mask("abnormal_robot")
        timeout_mask = self._termination_term_mask("time_out")
        path = self.failure_strat_export_path
        if not os.path.isabs(path) and self.log_dir is not None:
            path = os.path.join(self.log_dir, path)
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        raw_limit = getattr(self.env, "raw_action_limit", None)
        if torch.is_tensor(raw_limit):
            raw_limit = raw_limit.detach().to(device=self.device, dtype=self.dtype)

        with open(path, "a", encoding="utf-8") as f:
            for env_id_t in env_ids.detach().to(device="cpu").tolist():
                env_id = int(env_id_t)
                record = {
                    "label": self.failure_strat_label,
                    "it": int(it),
                    "step": int(step),
                    "env_id": env_id,
                    "success": None if success_mask is None else bool(success_mask[env_id].item()),
                    "terminated": bool(terms[env_id].detach().to(device="cpu").item()),
                    "truncated": bool(truncs[env_id].detach().to(device="cpu").item()),
                    "abnormal_robot": None if abnormal_mask is None else bool(abnormal_mask[env_id].item()),
                    "time_out": None if timeout_mask is None else bool(timeout_mask[env_id].item()),
                    "episode_return": self._record_scalar(rewards, env_id),
                    "episode_length": self._record_scalar(lengths, env_id),
                    "min_peg_in_hole_xyz": self._record_scalar(episode_min_xyz, env_id),
                    "init_peg_in_hole_xyz": self._record_vec(episode_start_obs, env_id, 0, 3),
                    "init_peg_in_hole_rot": self._record_vec(episode_start_obs, env_id, 3, 6),
                    "final_peg_in_hole_xyz": self._record_vec(final_policy, env_id, 0, 3),
                    "final_peg_in_hole_rot": self._record_vec(final_policy, env_id, 3, 6),
                    "init_raw_peg_in_hole_xyz": self._record_vec(episode_start_raw, env_id, 24, 27),
                    "final_raw_peg_in_hole_xyz": self._record_vec(final_raw, env_id, 24, 27),
                    "init_raw_ee_xyz": self._record_vec(episode_start_raw, env_id, 149, 152),
                    "init_raw_peg_xyz": self._record_vec(episode_start_raw, env_id, 179, 182),
                    "init_raw_hole_xyz": self._record_vec(episode_start_raw, env_id, 209, 212),
                    "final_raw_ee_xyz": self._record_vec(final_raw, env_id, 149, 152),
                    "final_raw_peg_xyz": self._record_vec(final_raw, env_id, 179, 182),
                    "final_raw_hole_xyz": self._record_vec(final_raw, env_id, 209, 212),
                    "init_gripper_mean": None,
                    "final_gripper_mean": None,
                    "teacher_policy_cos": None,
                    "teacher_policy_l2": None,
                    "exec_teacher_l2": None,
                    "exec_policy_l2": None,
                }
                init_gripper = self._record_vec(episode_start_raw, env_id, 113 + 6, 113 + 12)
                final_gripper = self._record_vec(final_raw, env_id, 113 + 6, 113 + 12)
                if init_gripper is not None:
                    record["init_gripper_mean"] = sum(init_gripper) / len(init_gripper)
                if final_gripper is not None:
                    record["final_gripper_mean"] = sum(final_gripper) / len(final_gripper)
                if torch.is_tensor(teacher_actions) and torch.is_tensor(policy_actions):
                    teacher = teacher_actions[env_id].detach()
                    policy = policy_actions[env_id].detach()
                    record["teacher_policy_cos"] = float(
                        torch.nn.functional.cosine_similarity(
                            teacher.view(1, -1), policy.view(1, -1), dim=-1
                        ).to(device="cpu").item()
                    )
                    record["teacher_policy_l2"] = float((teacher - policy).norm().to(device="cpu").item())
                    if torch.is_tensor(executed_actions):
                        executed = executed_actions[env_id].detach()
                        record["exec_teacher_l2"] = float((executed - teacher).norm().to(device="cpu").item())
                        record["exec_policy_l2"] = float((executed - policy).norm().to(device="cpu").item())
                    if torch.is_tensor(raw_limit):
                        record["teacher_policy_raw_l2"] = float(
                            ((teacher - policy) * raw_limit).norm().to(device="cpu").item()
                        )
                json.dump(record, f, sort_keys=True)
                f.write("\n")

    def _teacher_actions(self, *, ignore_alpha: bool = False) -> torch.Tensor | None:
        if self.teacher_policy is None:
            return None
        if not ignore_alpha and self._teacher_alpha() <= 0.0:
            return None
        raw_obs = getattr(self.env, "last_raw_policy_obs", None)
        raw_limit = getattr(self.env, "raw_action_limit", None)
        if raw_obs is None or raw_limit is None:
            return None
        raw_action = self.teacher_policy(raw_obs.to(device=self.device, dtype=self.dtype))
        raw_limit = raw_limit.to(device=self.device, dtype=self.dtype)
        return (raw_action / raw_limit).clamp(-1.0, 1.0)

    def _teacher_action_diag(
        self,
        teacher_actions: torch.Tensor,
        learner_actions: torch.Tensor,
        exec_actions: torch.Tensor,
        policy_mean_actions: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            eps = 1e-6
            teacher = teacher_actions.detach()
            learner = learner_actions.detach()
            executed = exec_actions.detach()
            policy_mean = policy_mean_actions.detach() if policy_mean_actions is not None else None
            cos = torch.nn.functional.cosine_similarity(teacher, learner, dim=-1)
            diff = learner - teacher
            raw_limit = getattr(self.env, "raw_action_limit", None)
            stats = {
                "TeacherDiag/alpha": self._teacher_alpha(),
                "TeacherDiag/learner_teacher_cos": cos.mean().item(),
                "TeacherDiag/learner_teacher_mae": diff.abs().mean().item(),
                "TeacherDiag/learner_teacher_l2": diff.norm(dim=-1).mean().item(),
                "TeacherDiag/teacher_norm": teacher.norm(dim=-1).mean().item(),
                "TeacherDiag/learner_norm": learner.norm(dim=-1).mean().item(),
                "TeacherDiag/exec_teacher_l2": (executed - teacher).norm(dim=-1).mean().item(),
            }
            if raw_limit is not None:
                scale = raw_limit.to(device=teacher.device, dtype=teacher.dtype)
                raw_diff = diff * scale
                stats["TeacherDiag/raw_mae"] = raw_diff.abs().mean().item()
                stats["TeacherDiag/raw_l2"] = raw_diff.norm(dim=-1).mean().item()
            if policy_mean is not None:
                pdiff = policy_mean - teacher
                lpdiff = learner - policy_mean
                stats["TeacherDiag/policy_teacher_cos"] = torch.nn.functional.cosine_similarity(
                    teacher, policy_mean, dim=-1
                ).mean().item()
                stats["TeacherDiag/policy_teacher_mae"] = pdiff.abs().mean().item()
                stats["TeacherDiag/planner_policy_l2"] = lpdiff.norm(dim=-1).mean().item()
                stats["TeacherDiag/policy_norm"] = policy_mean.norm(dim=-1).mean().item()
                if raw_limit is not None:
                    scale = raw_limit.to(device=teacher.device, dtype=teacher.dtype)
                    stats["TeacherDiag/policy_raw_mae"] = (pdiff * scale).abs().mean().item()
                    stats["TeacherDiag/planner_policy_raw_l2"] = (lpdiff * scale).norm(dim=-1).mean().item()
            return stats

    def _planner_action_ranking_diag(
        self,
        obs: Dict[str, torch.Tensor],
        teacher_actions: torch.Tensor,
        learner_actions: torch.Tensor,
        policy_mean_actions: torch.Tensor | None = None,
        extra_actions: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            planner = self.learner.planner
            rank_obs = obs
            if getattr(planner, "_obs_normalizer", None) is not None:
                rank_obs = planner._obs_normalizer(rank_obs)
            z0 = planner.encoder(rank_obs)

            def predict_next(action: torch.Tensor) -> torch.Tensor:
                z_traj = planner.dynamics(z0, action.unsqueeze(1))
                return z_traj[:, 1]

            def to_raw(next_z: torch.Tensor) -> torch.Tensor:
                if getattr(planner, "_obs_normalizer", None) is not None:
                    return planner._obs_normalizer.inverse({"policy": next_z})["policy"]
                return next_z

            def score_next(action: torch.Tensor, next_z: torch.Tensor) -> torch.Tensor:
                return planner.reward(z0, next_z, action=action)

            def q_stats(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                q_all = planner.value(z0, action, return_type="all")
                return q_all.mean(dim=0), q_all.std(dim=0)

            can_decode_progress = z0.shape[-1] == obs["policy"].shape[-1]
            current_dist = obs["policy"][..., 0:3].norm(dim=-1)

            def predicted_progress(next_z: torch.Tensor) -> torch.Tensor:
                if not can_decode_progress:
                    return torch.zeros_like(current_dist)
                next_raw = to_raw(next_z)
                return current_dist - next_raw[..., 0:3].norm(dim=-1)

            teacher_next = predict_next(teacher_actions.detach())
            learner_next = predict_next(learner_actions.detach())
            teacher_score = score_next(teacher_actions.detach(), teacher_next)
            learner_score = score_next(learner_actions.detach(), learner_next)
            teacher_q, teacher_q_std = q_stats(teacher_actions.detach())
            learner_q, learner_q_std = q_stats(learner_actions.detach())
            teacher_progress = predicted_progress(teacher_next)
            learner_progress = predicted_progress(learner_next)
            stats = {
                "PlannerDiag/reward_teacher": teacher_score.mean().item(),
                "PlannerDiag/reward_learner": learner_score.mean().item(),
                "PlannerDiag/reward_teacher_minus_learner": (
                    teacher_score - learner_score
                ).mean().item(),
                "PlannerDiag/reward_teacher_gt_learner": (
                    teacher_score > learner_score
                ).float().mean().item(),
                "PlannerDiag/q_teacher": teacher_q.mean().item(),
                "PlannerDiag/q_learner": learner_q.mean().item(),
                "PlannerDiag/q_teacher_minus_learner": (
                    teacher_q - learner_q
                ).mean().item(),
                "PlannerDiag/qstd_teacher": teacher_q_std.mean().item(),
                "PlannerDiag/qstd_learner": learner_q_std.mean().item(),
                "PlannerDiag/progress_skipped_latent_encoder": float(not can_decode_progress),
            }
            if can_decode_progress:
                stats.update({
                    "PlannerDiag/pred_progress_teacher": teacher_progress.mean().item(),
                    "PlannerDiag/pred_progress_learner": learner_progress.mean().item(),
                    "PlannerDiag/pred_progress_teacher_minus_learner": (
                        teacher_progress - learner_progress
                    ).mean().item(),
                    "PlannerDiag/pred_progress_teacher_gt_learner": (
                        teacher_progress > learner_progress
                    ).float().mean().item(),
                })
            if policy_mean_actions is not None:
                policy_next = predict_next(policy_mean_actions.detach())
                policy_score = score_next(policy_mean_actions.detach(), policy_next)
                policy_q, policy_q_std = q_stats(policy_mean_actions.detach())
                policy_progress = predicted_progress(policy_next)
                stats.update({
                    "PlannerDiag/reward_policy": policy_score.mean().item(),
                    "PlannerDiag/reward_teacher_minus_policy": (
                        teacher_score - policy_score
                    ).mean().item(),
                    "PlannerDiag/reward_learner_minus_policy": (
                        learner_score - policy_score
                    ).mean().item(),
                    "PlannerDiag/reward_teacher_gt_policy": (
                        teacher_score > policy_score
                    ).float().mean().item(),
                    "PlannerDiag/reward_learner_gt_policy": (
                        learner_score > policy_score
                    ).float().mean().item(),
                    "PlannerDiag/q_policy": policy_q.mean().item(),
                    "PlannerDiag/q_teacher_minus_policy": (
                        teacher_q - policy_q
                    ).mean().item(),
                    "PlannerDiag/q_learner_minus_policy": (
                        learner_q - policy_q
                    ).mean().item(),
                    "PlannerDiag/qstd_policy": policy_q_std.mean().item(),
                })
                if can_decode_progress:
                    stats.update({
                        "PlannerDiag/pred_progress_policy": policy_progress.mean().item(),
                        "PlannerDiag/pred_progress_teacher_minus_policy": (
                            teacher_progress - policy_progress
                        ).mean().item(),
                        "PlannerDiag/pred_progress_learner_minus_policy": (
                            learner_progress - policy_progress
                        ).mean().item(),
                        "PlannerDiag/pred_progress_teacher_gt_policy": (
                            teacher_progress > policy_progress
                        ).float().mean().item(),
                        "PlannerDiag/pred_progress_learner_gt_policy": (
                            learner_progress > policy_progress
                        ).float().mean().item(),
                    })
            for name, action in (extra_actions or {}).items():
                if not torch.is_tensor(action) or action.shape != learner_actions.shape:
                    continue
                if name == "gate_accept":
                    continue
                action = action.detach()
                next_z = predict_next(action)
                score = score_next(action, next_z)
                q_mean, q_std = q_stats(action)
                progress = predicted_progress(next_z)
                policy_next_action_score = None
                action_next_policy_score = None
                if policy_mean_actions is not None:
                    policy_next_action_score = score_next(action, policy_next)
                    action_next_policy_score = score_next(policy_mean_actions.detach(), next_z)
                stats.update({
                    f"PlannerDiag/reward_{name}": score.mean().item(),
                    f"PlannerDiag/reward_teacher_minus_{name}": (
                        teacher_score - score
                    ).mean().item(),
                    f"PlannerDiag/reward_{name}_minus_policy": (
                        score - policy_score
                    ).mean().item() if policy_mean_actions is not None else 0.0,
                    f"PlannerDiag/reward_teacher_gt_{name}": (
                        teacher_score > score
                    ).float().mean().item(),
                    f"PlannerDiag/q_{name}": q_mean.mean().item(),
                    f"PlannerDiag/q_teacher_minus_{name}": (
                        teacher_q - q_mean
                    ).mean().item(),
                    f"PlannerDiag/qstd_{name}": q_std.mean().item(),
                    f"PlannerDiag/reward_{name}Action_policyNext_minus_policy": (
                        policy_next_action_score - policy_score
                    ).mean().item() if policy_next_action_score is not None else 0.0,
                    f"PlannerDiag/reward_policyAction_{name}Next_minus_policy": (
                        action_next_policy_score - policy_score
                    ).mean().item() if action_next_policy_score is not None else 0.0,
                    f"PlannerDiag/{name}_teacher_cos": torch.nn.functional.cosine_similarity(
                        action, teacher_actions.detach(), dim=-1
                    ).mean().item(),
                    f"PlannerDiag/{name}_policy_cos": torch.nn.functional.cosine_similarity(
                        action, policy_mean_actions.detach(), dim=-1
                    ).mean().item() if policy_mean_actions is not None else 0.0,
                    f"PlannerDiag/{name}_teacher_l2": (
                        action - teacher_actions.detach()
                    ).norm(dim=-1).mean().item(),
                })
                if can_decode_progress:
                    stats.update({
                        f"PlannerDiag/pred_progress_{name}": progress.mean().item(),
                        f"PlannerDiag/pred_progress_teacher_minus_{name}": (
                            teacher_progress - progress
                        ).mean().item(),
                        f"PlannerDiag/pred_progress_{name}_minus_policy": (
                            progress - policy_progress
                        ).mean().item() if policy_mean_actions is not None else 0.0,
                        f"PlannerDiag/pred_progress_teacher_gt_{name}": (
                            teacher_progress > progress
                        ).float().mean().item(),
                        f"PlannerDiag/pred_progress_{name}_gt_policy": (
                            progress > policy_progress
                        ).float().mean().item() if policy_mean_actions is not None else 0.0,
                    })
            return stats

    def _dynamics_one_step_diag(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        if not isinstance(obs, dict) or "policy" not in obs or not isinstance(next_obs, dict) or "policy" not in next_obs:
            return {}
        with torch.no_grad():
            planner = self.learner.planner
            rank_obs = obs
            if getattr(planner, "_obs_normalizer", None) is not None:
                rank_obs = planner._obs_normalizer(rank_obs)
            z0 = planner.encoder(rank_obs)
            pred_next_z = planner.dynamics(z0, actions.detach().unsqueeze(1))[:, 1, :]
            target_next = next_obs["policy"].detach()
            if pred_next_z.shape[-1] != target_next.shape[-1]:
                target_obs = next_obs
                if getattr(planner, "_obs_normalizer", None) is not None:
                    target_obs = planner._obs_normalizer(target_obs)
                target_next_z = planner.encoder(target_obs)
                latent_err = pred_next_z - target_next_z
                return {
                    "DynDiag/latent_mse": latent_err.pow(2).mean().item(),
                    "DynDiag/latent_mae": latent_err.abs().mean().item(),
                    "DynDiag/raw_diag_skipped_latent_encoder": 1.0,
                }
            if getattr(planner, "_obs_normalizer", None) is not None:
                pred_next = planner._obs_normalizer.inverse({"policy": pred_next_z})["policy"]
            else:
                pred_next = pred_next_z
            err = pred_next - target_next

            def mse_slice(start: int, end: int) -> float:
                return err[..., start:end].pow(2).mean().item()

            def mae_slice(start: int, end: int) -> float:
                return err[..., start:end].abs().mean().item()

            stats = {
                "DynDiag/raw_mse": err.pow(2).mean().item(),
                "DynDiag/raw_mae": err.abs().mean().item(),
                "DynDiag/peg_in_hole_mse": mse_slice(0, 6),
                "DynDiag/peg_in_hole_mae": mae_slice(0, 6),
                "DynDiag/peg_xyz_mae": mae_slice(0, 3),
                "DynDiag/peg_delta_mse": mse_slice(6, 12),
                "DynDiag/ee_to_peg_mse": mse_slice(30, 36),
                "DynDiag/ee_to_peg_mae": mae_slice(30, 36),
                "DynDiag/ee_to_hole_mse": mse_slice(36, 42),
                "DynDiag/ee_to_hole_mae": mae_slice(36, 42),
                "DynDiag/gripper_mse": mse_slice(42, 48),
                "DynDiag/ee_delta_mse": mse_slice(48, 54),
                "DynDiag/prev_action_mse": mse_slice(60, 95) if target_next.shape[-1] >= 95 else 0.0,
                "DynDiag/joint_mse": mse_slice(95, 119) if target_next.shape[-1] >= 119 else 0.0,
            }
            actual_progress = obs["policy"][..., 0:3].norm(dim=-1) - target_next[..., 0:3].norm(dim=-1)
            pred_progress = obs["policy"][..., 0:3].norm(dim=-1) - pred_next[..., 0:3].norm(dim=-1)
            stats["DynDiag/progress_mae"] = (pred_progress - actual_progress).abs().mean().item()
            stats["DynDiag/actual_progress_mean"] = actual_progress.mean().item()
            stats["DynDiag/pred_progress_mean"] = pred_progress.mean().item()
            stats["DynDiag/actual_progress_pos_rate"] = (actual_progress > 0.0).float().mean().item()
            stats["DynDiag/pred_progress_pos_rate"] = (pred_progress > 0.0).float().mean().item()
            stats["DynDiag/progress_sign_acc"] = (
                torch.sign(pred_progress) == torch.sign(actual_progress)
            ).float().mean().item()
            return stats

    def _action_stats(self, actions: torch.Tensor, env_actions: torch.Tensor, infos: Any) -> Dict[str, float]:
        with torch.no_grad():
            stats = {
                "Action/norm": actions.norm(dim=-1).mean().item(),
                "Action/abs_mean": actions.abs().mean().item(),
                "Action/abs_max": actions.abs().max().item(),
                "Action/env_norm": env_actions.norm(dim=-1).mean().item(),
                "Action/env_abs_mean": env_actions.abs().mean().item(),
                "Action/env_abs_max": env_actions.abs().max().item(),
            }
            for i in range(actions.shape[-1]):
                stats[f"Action/dim{i}_mean"] = actions[..., i].mean().item()
                stats[f"Action/dim{i}_std"] = actions[..., i].std(unbiased=False).item()
                stats[f"Action/dim{i}_abs_mean"] = actions[..., i].abs().mean().item()
                stats[f"Action/dim{i}_clip_ratio"] = (actions[..., i].abs() >= 0.999).float().mean().item()

            if isinstance(infos, dict):
                raw_action = infos.get("mpail_omnireset/raw_action")
                if torch.is_tensor(raw_action):
                    raw_action = raw_action.detach()
                    stats.update({
                        "Action/raw_norm": raw_action.norm(dim=-1).mean().item(),
                        "Action/raw_abs_mean": raw_action.abs().mean().item(),
                        "Action/raw_abs_max": raw_action.abs().max().item(),
                    })
                    for i in range(raw_action.shape[-1]):
                        stats[f"Action/raw_dim{i}_mean"] = raw_action[..., i].mean().item()
                        stats[f"Action/raw_dim{i}_std"] = raw_action[..., i].std(unbiased=False).item()
                        stats[f"Action/raw_dim{i}_abs_mean"] = raw_action[..., i].abs().mean().item()
                        raw_limit = getattr(self.env, "raw_action_limit", None)
                        if raw_limit is not None:
                            limit = raw_limit.to(device=raw_action.device, dtype=raw_action.dtype).view(-1)[i]
                            stats[f"Action/raw_dim{i}_clip_ratio"] = (
                                raw_action[..., i].abs() >= (0.999 * limit)
                            ).float().mean().item()
            return stats
