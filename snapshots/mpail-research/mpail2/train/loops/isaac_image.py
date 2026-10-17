"""Isaac Lab image training loop: env first, learner from observation spaces, then runner."""

from __future__ import annotations

import os
import traceback
import importlib
from typing import Any

import gymnasium as gym
import torch

from mpail2.envs.isaac.launcher import isaac_session
from mpail2.envs.isaac.omnireset_wrapper import (
    COMPACT_OBS_DIM,
    COMPACT_OBS_WITH_EE_VEL_DIM,
    DEFAULT_RAW_ACTION_LIMIT,
    PLANNER_OBS_V2_DIM,
    PLANNER_OBS_V3_DIM,
    OmniResetMPAILWrapper,
    compact_omnireset_demonstrations,
)
from mpail2.envs.utils import CustomRecordVideo
from mpail2.runner import MPAIL2Runner

from mpail2.train.registry import TrainEnvSpec
from mpail2.train.utils.demos import load_demonstrations
from mpail2.train.utils.hydra_learner import learner_config_from_train_cfg
from mpail2.train.utils.isaac_runner_assembly import assemble_isaac_runner
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env
from mpail2.train.utils.runtime import ensure_wandb_settings, finish_wandb_if_running


def _apply_omnireset_reset_overrides(env_cfg: Any, cfg: Any) -> None:
    reset_types = getattr(cfg, "omnireset_reset_types", None)
    reset_probs = getattr(cfg, "omnireset_reset_probs", None)
    if not reset_types:
        return

    reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
    params = getattr(reset_event, "params", None)
    if params is None:
        raise ValueError("Requested omnireset_reset_types, but env_cfg has no reset_from_reset_states params.")

    params["reset_types"] = list(reset_types)
    params["probs"] = list(reset_probs) if reset_probs else [1.0 / len(reset_types)] * len(reset_types)
    print(f"[INFO] OmniReset reset override: reset_types={params['reset_types']} probs={params['probs']}")


def _apply_omnireset_termination_overrides(env_cfg: Any, cfg: Any) -> None:
    if not getattr(cfg, "omnireset_disable_abnormal_termination", False):
        return

    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None or not hasattr(terminations, "abnormal_robot"):
        raise ValueError("Requested omnireset_disable_abnormal_termination, but env_cfg has no abnormal_robot term.")

    terminations.abnormal_robot = None
    print("[INFO] OmniReset termination override: disabled abnormal_robot; kept time_out enabled.")


def run_isaac_image_training(cfg: Any, launcher_args, spec: TrainEnvSpec) -> None:
    """OmegaConf or dataclass cfg; must expose .task, .log, .runner, .device, etc."""
    ensure_wandb_settings(cfg.log)

    with isaac_session(
        launcher_args,
        enable_cameras=cfg.enable_cameras,
        log_level=cfg.log.log_level,
    ):
        try:
            import mpail2.envs.isaac.franka  # noqa: F401

            for module_name in spec.import_modules:
                importlib.import_module(module_name)
            from isaaclab_tasks.utils import parse_env_cfg

            log_cfg = cfg.log

            env_cfg = parse_env_cfg(cfg.task, device=cfg.device, num_envs=cfg.num_envs)
            _apply_omnireset_reset_overrides(env_cfg, cfg)
            _apply_omnireset_termination_overrides(env_cfg, cfg)
            render_mode = "rgb_array" if log_cfg.video else None
            env = gym.make(cfg.task, cfg=env_cfg, render_mode=render_mode)
            if getattr(cfg, "omnireset_mpail_wrapper", False):
                raw_action_limit = getattr(cfg, "omnireset_raw_action_limit", None) or list(DEFAULT_RAW_ACTION_LIMIT)
                compact_obs_dim = getattr(cfg, "omnireset_compact_obs_dim", COMPACT_OBS_DIM)
                env = OmniResetMPAILWrapper(
                    env,
                    raw_action_limit=raw_action_limit,
                    compact_obs_dim=compact_obs_dim,
                    require_held_reset=getattr(cfg, "omnireset_require_held_reset", False),
                    held_distance_threshold=getattr(cfg, "omnireset_held_distance_threshold", 0.08),
                    max_held_reset_attempts=getattr(cfg, "omnireset_max_held_reset_attempts", 100),
                    force_gripper_close=getattr(cfg, "omnireset_force_gripper_close", False),
                    gripper_close_value=getattr(cfg, "omnireset_gripper_close_value", -1.0),
                )
                print(
                    "[INFO] Enabled OmniReset MPAIL wrapper: "
                    f"compact_obs={compact_obs_dim} raw_action_limit={list(raw_action_limit)} "
                    f"require_held_reset={getattr(cfg, 'omnireset_require_held_reset', False)} "
                    f"held_threshold={getattr(cfg, 'omnireset_held_distance_threshold', 0.08)} "
                    f"force_gripper_close={getattr(cfg, 'omnireset_force_gripper_close', False)}"
                )

            learner_cfg = learner_config_from_train_cfg(cfg)
            if cfg.sync_learner_dims_from_env:
                learner_cfg = sync_learner_dims_from_env(learner_cfg, env)
            runner_cfg = assemble_isaac_runner(
                cfg,
                learner_cfg,
                demo_env_var=spec.demo_env_var,
                default_demo_rel=spec.default_demo_rel,
                default_num_iterations=spec.default_num_iterations,
            )

            if log_cfg.video:
                if log_cfg.video_length:
                    vid_length = log_cfg.video_length
                else:
                    vid_length = env.unwrapped.max_episode_length

                video_kwargs = {
                    "video_folder": os.path.join(log_cfg.run_log_dir, "videos"),
                    "step_trigger": lambda step: step % log_cfg.video_interval == 0,
                    "video_length": vid_length,
                    "disable_logger": True,
                    "enable_wandb": not log_cfg.no_wandb,
                    "video_resolution": log_cfg.video_resolution,
                    "video_crf": log_cfg.video_crf,
                }
                print("[INFO] Recording videos during training.")
                env = CustomRecordVideo(env, **video_kwargs)

            demonstrations = load_demonstrations(runner_cfg.path_to_demonstrations, device=cfg.device)
            if getattr(cfg, "omnireset_mpail_wrapper", False):
                compact_obs_dim = getattr(cfg, "omnireset_compact_obs_dim", COMPACT_OBS_DIM)
                if compact_obs_dim != 215:
                    demo_mode = "planner_v3" if compact_obs_dim == PLANNER_OBS_V3_DIM else (
                        "planner_v2" if compact_obs_dim == PLANNER_OBS_V2_DIM else
                        compact_obs_dim in (COMPACT_OBS_DIM, COMPACT_OBS_WITH_EE_VEL_DIM)
                    )
                    demonstrations = compact_omnireset_demonstrations(
                        demonstrations,
                        include_ee_to_hole=demo_mode,
                        include_ee_vel=compact_obs_dim == COMPACT_OBS_WITH_EE_VEL_DIM,
                    )
                if "policy" in demonstrations:
                    print(f"[INFO] Compact OmniReset demos: policy shape={tuple(demonstrations['policy'].shape)}")

            runner = MPAIL2Runner(
                demonstrations=demonstrations,
                env=env,
                runner_cfg=runner_cfg,
                device=cfg.device,
                dtype=torch.float32,
            )
            try:
                runner.learn()
            finally:
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    finish_wandb_if_running(log_cfg=log_cfg)
                except Exception:
                    pass

        except Exception:
            print("\n[ERROR] Exception occurred in training script:")
            traceback.print_exc()
            raise
