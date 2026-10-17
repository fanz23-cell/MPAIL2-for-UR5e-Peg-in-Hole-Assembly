"""Meta-World image training loop: Hydra config, multi-camera env, then runner."""

from __future__ import annotations

import dataclasses
import os
import random
from datetime import datetime
from typing import Any

import torch
from omegaconf import OmegaConf

from mpail2.envs.gym_mujoco.utils import load_demonstrations
from mpail2.envs.metaworld.env_factory import setup_metaworld_environment
from mpail2.runner import MPAIL2Runner
from mpail2.train.configs.metaworld_train_config import MetaworldHydraTrainConfig
from mpail2.train.configs.metaworld_learner_config import (
    MetaworldMT1ImageLearnerConfig,
    MetaworldMT1ImageVisLearnerConfig,
    metaworld_mt1_image_planner_config,
    metaworld_mt1_image_vis_planner_config,
)
from mpail2.utils.image_rollout_vis import ImageSpaceRolloutsVisualization
from mpail2.train.gym_tasks.gym_assembly import (
    apply_gym_learner_overrides,
    build_gym_runner_cfg,
)
from mpail2.train.registry import TrainEnvSpec
from mpail2.train.utils.demos import resolve_metaworld_demo_path
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env
from mpail2.train.utils.runtime import (
    ensure_wandb_settings,
    finish_wandb_if_running,
    make_runner_log_cfg,
    maybe_disable_wandb_if_unavailable,
    maybe_init_wandb,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _get_mujoco_model_from_env(env):
    """Traverse the wrapper chain to find the MuJoCo model of the first sub-env.

    Expected chain (inner → outer):
        SawyerXYZEnv  →  MetaWorldMPAILEnv  →  SyncVectorEnv
        →  MetaWorldVectorTensorWrapper  [→  VideoRecordingWrapper]

    Returns the ``mujoco.MjModel`` or ``None`` if not found.
    """
    candidate = env
    while candidate is not None:
        envs_list = getattr(candidate, "envs", None)
        if envs_list and len(envs_list) > 0:
            sub = envs_list[0]
            unwrapped = getattr(sub, "unwrapped", sub)
            if hasattr(unwrapped, "model"):
                return unwrapped.model
        candidate = getattr(candidate, "env", None)
    return None


def run_metaworld_training(cfg: Any, spec: TrainEnvSpec, *, resume_from: str | None = None) -> None:
    """Run MPAIL on Meta-World from a Hydra-composed config."""
    os.environ.setdefault("MUJOCO_GL", "egl")

    _raw_cfg = cfg  # keep OmegaConf ref for config.yaml saving
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_object(cfg)
    if not isinstance(cfg, MetaworldHydraTrainConfig):
        raise TypeError(f"Expected MetaworldHydraTrainConfig, got {type(cfg)}")

    cameras = tuple(str(c).strip() for c in cfg.cameras if str(c).strip())
    if not cameras:
        raise ValueError("cfg.cameras must be a non-empty list of camera names")

    demo_path = resolve_metaworld_demo_path(
        cfg.demo_path,
        spec.demo_env_var,
        spec.default_demo_rel,
    )

    run_name = cfg.log.run_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    elif resume_from is None:
        run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg.log.run_name = run_name

    safe_env = cfg.env_id.replace("/", "_").replace(" ", "_")
    log_dir = os.path.join(cfg.log.run_log_dir, safe_env, run_name)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)

    # Save config (before mutating run_log_dir) and meta for --continue support.
    import json as _json
    if OmegaConf.is_config(_raw_cfg):
        OmegaConf.save(_raw_cfg, os.path.join(log_dir, "config.yaml"))
    with open(os.path.join(log_dir, "meta.json"), "w") as _f:
        _json.dump(
            {
                "suite": spec.suite,
                "env_id": cfg.env_id,
                "run_name": run_name,
                "config_name": spec.config_name,
                "demo_env_var": spec.demo_env_var,
                "default_demo_rel": spec.default_demo_rel,
                "default_num_iterations": spec.default_num_iterations,
            },
            _f,
            indent=2,
        )

    cfg.log.run_log_dir = log_dir

    video_dir = os.path.join(log_dir, "videos") if cfg.log.video else None

    print("=" * 60)
    print("MPAIL on Meta-World (Hydra)")
    print("=" * 60)
    print(f"Demo: {demo_path}")
    print(f"env_id: {cfg.env_id}  num_envs: {cfg.num_envs}  device: {cfg.device}")
    print(f"cameras: {cameras}  images: {cfg.image_width}x{cfg.image_height}")
    print(f"Log: {log_dir}")
    print("=" * 60)

    random.seed(cfg.runner.seed)
    torch.manual_seed(cfg.runner.seed)

    use_wandb = maybe_disable_wandb_if_unavailable(cfg.log, wandb_available=WANDB_AVAILABLE)
    if use_wandb:
        ensure_wandb_settings(cfg.log)
    maybe_init_wandb(
        log_cfg=cfg.log,
        run_name=run_name,
        config_obj=dataclasses.asdict(cfg),
        run_dir=log_dir,
    )
    if use_wandb:
        print(
            f"[INFO] Wandb logging enabled: {cfg.log.wandb_entity}/{cfg.log.wandb_project}/{run_name}"
        )

    env = setup_metaworld_environment(
        env_id=cfg.env_id,
        num_envs=cfg.num_envs,
        max_episode_length=cfg.max_episode_length,
        device=cfg.device,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
        camera_names=cameras,
        flip_rgb_vertical=cfg.flip_rgb_vertical,
        use_goal_hidden=cfg.use_goal_hidden,
        include_object_state=cfg.include_object_state,
        video_folder=video_dir,
        video_step_trigger=lambda step: step % cfg.log.video_interval == 0,
        video_length=cfg.log.video_length or cfg.max_episode_length,
        enable_wandb=use_wandb,
        video_fps=cfg.log.video_fps,
    )

    demonstrations, demo_meta = load_demonstrations(
        demo_path, device=cfg.device, num_demos=cfg.num_demos
    )
    for k, t in demonstrations.items():
        print(f"  demo {k}: {tuple(t.shape)}")
    if demo_meta:
        print(f"  demo metadata keys: {list(demo_meta.keys())}")

    if getattr(cfg, "vis_rollouts", False):
        # Image-space rollout vis: proprioception decoder + corner-camera projection.
        learner_cfg = MetaworldMT1ImageVisLearnerConfig(
            planner_cfg=metaworld_mt1_image_vis_planner_config(
                image_h=cfg.image_height,
                image_w=cfg.image_width,
                camera_names=cameras,
                include_object_state_encoder=cfg.include_object_state,
            )
        )
        print("[INFO] Using image-space rollout visualisation (MetaworldMT1ImageVisLearnerConfig).")
    else:
        learner_cfg = MetaworldMT1ImageLearnerConfig(
            planner_cfg=metaworld_mt1_image_planner_config(
                image_h=cfg.image_height,
                image_w=cfg.image_width,
                camera_names=cameras,
                include_object_state_encoder=cfg.include_object_state,
            )
        )
    apply_gym_learner_overrides(learner_cfg, cfg.learner)
    if cfg.sync_learner_dims_from_env:
        learner_cfg = sync_learner_dims_from_env(learner_cfg, env)

    # Enable vis-rollout video capture when the planner has a vis_cfg.
    has_vis_cfg = getattr(learner_cfg.planner_cfg, "vis_cfg", None) is not None

    runner_cfg = build_gym_runner_cfg(
        learner_cfg=learner_cfg,
        num_learning_iterations=cfg.runner.num_learning_iterations,
        path_to_demonstrations=os.path.abspath(demo_path),
        seed=cfg.runner.seed,
        logger="wandb" if use_wandb else None,
        log_cfg=make_runner_log_cfg(cfg.log, logger="wandb" if use_wandb else None),
        vis_rollouts=has_vis_cfg and use_wandb,
    )

    if use_wandb:
        import wandb as wandb_mod

        wandb_mod.config.update(dataclasses.asdict(runner_cfg))

    runner = MPAIL2Runner(
        demonstrations=demonstrations,
        env=env,
        runner_cfg=runner_cfg,
        device=cfg.device,
    )
    env.seed(cfg.runner.seed)

    # For image-space vis: extract MuJoCo camera intrinsics/extrinsics from
    # the first sub-environment's model and cache them on the visualiser.
    vis = runner.learner.planner.vis
    if isinstance(vis, ImageSpaceRolloutsVisualization):
        mj_model = _get_mujoco_model_from_env(env)
        if mj_model is not None:
            vis.setup_camera(mj_model, vis.cfg.camera_name)
        else:
            print(
                "[WARN] Could not locate a MuJoCo model in the env wrapper chain. "
                "Camera intrinsics will not be set; rollout projection will be skipped."
            )

    if resume_from is not None:
        print(f"[INFO] Resuming from checkpoint: {resume_from}")
        runner.load(resume_from)

    print("[INFO] Starting training...")
    try:
        runner.learn()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        runner.save(postfix="final")
        finish_wandb_if_running(log_cfg=cfg.log)
        env.close()
    print(f"[INFO] Done. Models under {os.path.join(log_dir, 'models')}")
