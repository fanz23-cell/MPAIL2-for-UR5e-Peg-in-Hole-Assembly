#!/usr/bin/env python3
"""Replay saved OmniReset raw rollout actions in the Isaac Sim viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Watch saved OmniReset expert rollout actions.")
parser.add_argument("--rollouts", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to replay; 0 means all.")
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--print-every", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--reset-type", default=None)
parser.add_argument("--clip-actions", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402


def _progress_term(env):
    return env.unwrapped.reward_manager.get_term_cfg("progress_context").func


def main() -> None:
    device = args_cli.device or "cuda"
    bundle = torch.load(str(args_cli.rollouts), map_location="cpu", weights_only=False)
    data = bundle["data"]
    metadata = bundle.get("metadata", {})
    episode_ids = data["episode_id"].long()
    actions = data["actions"].float()
    saved_success = data.get("success")
    saved_xyz = data.get("xyz_distance")
    saved_rewards = data.get("rewards")

    unique_ids = torch.unique_consecutive(episode_ids).tolist()
    if args_cli.episodes and args_cli.episodes > 0:
        unique_ids = unique_ids[: args_cli.episodes]

    reset_type = args_cli.reset_type
    if reset_type is None:
        reset_type = metadata.get("reset_type")
    print(
        f"[INFO] Replaying {len(unique_ids)} saved episodes from {args_cli.rollouts} "
        f"reset_type={reset_type!r}",
        flush=True,
    )
    print(f"[INFO] Metadata: {metadata}", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env_cfg.seed = args_cli.seed
    if reset_type:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested reset type, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = [reset_type]
        params["probs"] = [1.0]
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        for ep_idx, ep_id in enumerate(unique_ids, start=1):
            if not simulation_app.is_running():
                break
            mask = episode_ids == int(ep_id)
            indices = mask.nonzero(as_tuple=False).flatten()
            ep_actions = actions[indices].to(device=device)
            obs, _ = env.reset()
            total_reward = 0.0
            print(
                f"[EPISODE_START] {ep_idx}/{len(unique_ids)} saved_episode_id={int(ep_id)} "
                f"saved_steps={len(indices)} saved_success_any={bool(saved_success[indices].any()) if saved_success is not None else '?'}",
                flush=True,
            )
            for step, action in enumerate(ep_actions):
                if not simulation_app.is_running():
                    break
                env_action = action.view(1, -1)
                if args_cli.clip_actions and args_cli.clip_actions > 0:
                    env_action = env_action.clamp(-args_cli.clip_actions, args_cli.clip_actions)
                obs, rewards, terms, truncs, _infos = env.step(env_action.to(env.unwrapped.device))
                done = bool((terms | truncs).any().detach().cpu())
                total_reward += float(rewards.mean().detach().cpu())
                progress = _progress_term(env)
                live_success = bool(progress.success.detach().bool().any().cpu())
                live_xyz = float(progress.xyz_distance.detach().float().mean().cpu())
                if step % args_cli.print_every == 0 or done or step == len(ep_actions) - 1:
                    saved_s = bool(saved_success[indices[step]]) if saved_success is not None else None
                    saved_d = float(saved_xyz[indices[step]]) if saved_xyz is not None else None
                    saved_r = float(saved_rewards[indices[step]]) if saved_rewards is not None else None
                    print(
                        f"[EP {ep_idx:02d}/{len(unique_ids):02d}] step={step:03d}/{len(ep_actions)} "
                        f"live_reward={float(rewards.mean().detach().cpu()):.4f} total={total_reward:.3f} "
                        f"live_success={live_success} live_xyz={live_xyz:.4f} "
                        f"saved_success={saved_s} saved_xyz={saved_d} saved_reward={saved_r} done={done}",
                        flush=True,
                    )
                if done:
                    break
                if args_cli.sleep > 0:
                    time.sleep(args_cli.sleep)
            print(f"[EPISODE_END] {ep_idx}/{len(unique_ids)} saved_episode_id={int(ep_id)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
