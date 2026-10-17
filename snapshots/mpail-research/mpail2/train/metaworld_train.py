"""Run MPAIL on Meta-World (dict image + proprio observations).

Prefer the unified Hydra entry::

    MUJOCO_GL=egl python -m mpail2.train.train --env mw/basketball-v3 \\
      demo_path=demos/mw_basketball-v3/demos.pt log.no_wandb=True

With wandb (project/entity on the CLI)::

    python -m mpail2.train.train --env mw/basketball-v3 demo_path=... \\
      log.no_wandb=false log.wandb_project=my_proj log.wandb_entity=my_team

This module keeps a legacy argparse CLI that delegates to the same training loop.
"""

from __future__ import annotations

import argparse

from mpail2.envs.metaworld.mw_obs import DEFAULT_CAMERAS
from mpail2.train.configs.gym_train_config import GymLogHydraCfg, GymRunnerHydraCfg
from mpail2.train.configs.metaworld_train_config import MetaworldHydraTrainConfig
from mpail2.train.loops.metaworld_image import run_metaworld_training
from mpail2.train.registry import resolve_train_env


def main() -> None:
    p = argparse.ArgumentParser(description="MPAIL training on Meta-World image+proprio demos.")
    p.add_argument("--demo", type=str, required=True, help="Path to .pt from data_collection")
    p.add_argument("--env-id", type=str, required=True, help="e.g. mw/basketball-v3 (must match demo task)")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--max-episode-length", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-demos", type=int, default=None)
    p.add_argument("--image-width", type=int, default=84)
    p.add_argument("--image-height", type=int, default=84)
    p.add_argument(
        "--cameras",
        type=str,
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated; must match collection",
    )
    p.add_argument(
        "--no-flip-rgb-vertical",
        action="store_true",
        help="Disable vertical flip for all cameras (default: per-camera; corner flipped, gripperPOV not)",
    )
    p.add_argument("--use-goal-hidden", action="store_true")
    p.add_argument("--log-root", type=str, default="./logs/mpail/metaworld")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--video", action="store_true", help="Record rollout videos under the run log dir")
    p.add_argument("--video-interval", type=int, default=5000)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--wandb", action="store_true", help="Enable wandb")
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project (or set WANDB_PROJECT / use Hydra log.wandb_project=...)",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity/team (or set WANDB_ENTITY / use Hydra log.wandb_entity=...)",
    )
    args = p.parse_args()

    cams = [s.strip() for s in args.cameras.split(",") if s.strip()]
    log_cfg = GymLogHydraCfg(
        run_log_dir=args.log_root,
        run_name=args.run_name,
        no_wandb=not args.wandb,
        checkpoint_every=args.checkpoint_every,
        video=args.video,
        video_interval=args.video_interval,
        video_length=args.max_episode_length,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    runner_cfg = GymRunnerHydraCfg(num_learning_iterations=args.num_iterations, seed=args.seed)

    cfg = MetaworldHydraTrainConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        max_episode_length=args.max_episode_length,
        device=args.device,
        demo_path=args.demo,
        num_demos=args.num_demos,
        image_width=args.image_width,
        image_height=args.image_height,
        cameras=cams,
        flip_rgb_vertical=False if args.no_flip_rgb_vertical else None,
        use_goal_hidden=args.use_goal_hidden,
        log=log_cfg,
        runner=runner_cfg,
    )
    spec = resolve_train_env("metaworld")
    run_metaworld_training(cfg, spec)


if __name__ == "__main__":
    main()
