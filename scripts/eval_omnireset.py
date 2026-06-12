#!/usr/bin/env python3
"""Evaluate a trained MPAIL2 checkpoint on OmniReset peg-in-hole and record a video.

Usage:
    conda run -n mpail2-omnireset python scripts/eval_omnireset.py \
        --checkpoint logs/models/model_50.pt \
        --output /tmp/eval_iter50.mp4 \
        --num-envs 1 --num-episodes 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("/tmp/eval_omnireset.mp4"))
parser.add_argument(
    "--demo-path",
    type=Path,
    default=Path("mpail2/train/isaac_franka/demos/peg_expert_rl_mpail_filtered.pt"),
    help="MPAIL demo file used to reconstruct the fixed observation normalizer.",
)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--num-episodes", type=int, default=3)
parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="Optional hard cap on total eval steps; useful for quick videos.",
)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import numpy as np
import imageio

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa
import uwlab_tasks     # noqa

from mpail2.runner import MPAIL2Runner
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig
from mpail2.train.utils.demos import load_demonstrations
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env
from mpail2.learner import MPAIL2Learner


def main():
    device = args_cli.device or "cuda"

    # Build env with rendering
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # Build learner
    learner_cfg = OmniResetStateLearnerConfig()
    learner_cfg = sync_learner_dims_from_env(learner_cfg, env)

    # Use the same demo statistics as training for the fixed observation normalizer.
    demonstrations = load_demonstrations(str(args_cli.demo_path), device=device)
    obs_dim = learner_cfg.planner_cfg.encoder_cfg.input_dim
    learner = MPAIL2Learner(demonstrations, args_cli.num_envs, learner_cfg, device=device)
    learner.init_storage(
        num_envs=args_cli.num_envs,
        num_steps_per_env=env.unwrapped.max_episode_length,
        actor_obs_shape={"policy": (obs_dim,)},
        critic_obs_shape=None,
        action_shape=[learner_cfg.planner_cfg.action_dim],
    )

    # Load checkpoint
    ckpt = torch.load(str(args_cli.checkpoint), map_location=device, weights_only=False)
    learner.planner.load_state_dict(ckpt["model_state_dict"])
    learner.planner.eval()
    print(f"[INFO] Loaded checkpoint iter={ckpt.get('iter','?')}")

    # Rollout and record
    frames = []
    obs, _ = env.reset()
    ep_reward = 0.0
    episodes_done = 0
    default_max_steps = env.unwrapped.max_episode_length * args_cli.num_episodes + 10
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else default_max_steps

    for step in range(max_steps):
        with torch.inference_mode():
            actions = learner.act(obs)

        obs, rewards, terms, truncs, infos = env.step(actions)
        ep_reward += rewards.sum().item()

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        dones = (terms | truncs)
        if dones.any():
            episodes_done += 1
            print(f"  Episode {episodes_done} done at step {step}, total_reward={ep_reward:.3f}")
            ep_reward = 0.0
            if episodes_done >= args_cli.num_episodes:
                break
            obs, _ = env.reset()

    env.close()

    if frames:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(args_cli.output), frames, fps=10)
        print(f"Video saved → {args_cli.output}  ({len(frames)} frames)")
    else:
        print("[WARN] No frames captured — render may not be supported in headless mode")


if __name__ == "__main__":
    main()
    simulation_app.close()
