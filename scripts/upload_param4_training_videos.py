#!/usr/bin/env python3
"""Upload checkpoint rollout videos for the obs22 parameter sweep to W&B."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--project", default="mpail2-omnireset")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = sorted(args.base.glob("E*/checkpoint_videos/*.mp4"))
    if not videos:
        raise SystemExit(f"No videos found under {args.base}/E*/checkpoint_videos")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type="checkpoint_video_upload",
        config={
            "source_base": str(args.base),
            "purpose": "Training-process checkpoint videos for Tyler diagnostics",
            "checkpoints": [0, 50, 100, 250, 500],
            "runs": ["E1_replay_ratio", "E7_reward_lr1e-4"],
        },
    )

    for video in videos:
        run_name = video.parent.parent.name
        stem = video.stem
        iter_token = stem.split("_model_")[-1]
        try:
            step = int(iter_token)
        except ValueError:
            step = None
        key = f"checkpoint_videos/{run_name}/{stem}"
        wandb.log({key: wandb.Video(str(video), fps=args.fps, format="mp4")}, step=step)
        print(f"[wandb] uploaded {video}")

    run.finish()


if __name__ == "__main__":
    main()
