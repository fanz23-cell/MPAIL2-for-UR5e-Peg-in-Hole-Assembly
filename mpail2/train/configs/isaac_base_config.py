"""Shared Hydra dataclasses for Isaac image training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class LogConfig:
    run_log_dir: str = "./logs"
    run_name: str = "mpail2_isaac"
    no_wandb: bool = False
    video: bool = True
    video_interval: int = 100
    video_length: Optional[int] = None
    video_resolution: tuple = (640, 360)
    video_crf: int = 30
    checkpoint_every: int = 10
    model_save_dirname: str = "models"
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    log_level: str = "error"


@dataclass
class IsaacRunnerCfg:
    path_to_demonstrations: Optional[str] = None
    num_learning_iterations: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    resume_partial_ignore_mismatch: bool = False
    action_scale: Optional[list[float]] = None
    teacher_policy_path: Optional[str] = None
    teacher_bootstrap_steps: int = 0
    teacher_blend_steps: int = 0
    policy_control_after_online_teacher_bc: bool = False
    teacher_diag_always: bool = False
    teacher_dagger_collect: bool = False
    teacher_dagger_buffer_size: int = 0
    teacher_dagger_export_path: Optional[str] = None
    teacher_dagger_label_mode: str = "teacher"
    teacher_intervention_cos_threshold: Optional[float] = None
    teacher_residual_beta: float = 0.0
    teacher_residual_final_beta: Optional[float] = None
    teacher_oracle_mppi_gate: bool = False
    teacher_oracle_mppi_pool_select: bool = False
    teacher_oracle_mppi_margin: float = 0.0
    failure_strat_export_path: Optional[str] = None
    failure_strat_label: str = "policy"
    seed: Optional[int] = 0
    logger: Optional[str] = None
    vis_rollouts: bool = False
    render_during_training: bool = False
    render_interval_steps: int = 1


@dataclass
class IsaacHydraTrainConfig:
    """Base fields shared by Isaac image envs; subclasses set task and runner defaults."""

    task: str = ""
    num_envs: int = 1
    device: str = "cuda"
    log: LogConfig = field(default_factory=LogConfig)
    runner: IsaacRunnerCfg = field(default_factory=IsaacRunnerCfg)
    # Optional CLI-tunable overrides applied onto task-specific learner presets.
    learner: dict[str, Any] = field(default_factory=dict)
    omnireset_reset_types: Optional[list[str]] = None
    omnireset_reset_probs: Optional[list[float]] = None
    omnireset_disable_abnormal_termination: bool = False
    omnireset_mpail_wrapper: bool = False
    omnireset_compact_obs_dim: int = 19
    omnireset_raw_action_limit: Optional[list[float]] = None
    omnireset_require_held_reset: bool = False
    omnireset_held_distance_threshold: float = 0.08
    omnireset_max_held_reset_attempts: int = 100
    omnireset_force_gripper_close: bool = False
    omnireset_gripper_close_value: float = -1.0
    headless: bool = True
    enable_cameras: bool = True
    # If True, patch planner action_dim and encoder shapes from the live env after gym.make.
    sync_learner_dims_from_env: bool = True
