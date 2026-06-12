#!/usr/bin/env python3
"""Append one video artifact/metric to an existing W&B run."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="mpail2-omnireset")
    parser.add_argument("--entity", default="fanz23-university-of-washington")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if not args.video.exists() or args.video.stat().st_size == 0:
        raise FileNotFoundError(args.video)

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="allow",
        job_type="append_training_process_video",
    )
    wandb.log(
        {
            args.key: wandb.Video(str(args.video), fps=args.fps, format="mp4"),
            f"{args.key}_path": str(args.video),
        },
        step=args.step,
    )
    run.finish()
    print(f"[wandb] uploaded {args.video} to {args.entity}/{args.project}/runs/{args.run_id}")


if __name__ == "__main__":
    main()
