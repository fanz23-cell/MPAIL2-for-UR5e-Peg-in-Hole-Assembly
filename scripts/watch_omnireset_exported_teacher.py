#!/usr/bin/env python3
"""Watch the exported OmniReset teacher policy used by the demo collector."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Watch exported TorchScript OmniReset teacher policy live.")
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--print-every", type=int, default=10)
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--clip-actions", type=float, default=0.0)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--reset-type", default="ObjectAnywhereEEAnywhere")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402


def _policy_obs(obs) -> torch.Tensor:
    return obs if isinstance(obs, torch.Tensor) else obs["policy"]


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env_cfg.seed = args_cli.seed
    if args_cli.reset_type:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested reset type, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = [args_cli.reset_type]
        params["probs"] = [1.0]
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    policy = torch.jit.load(str(args_cli.policy), map_location=device).eval()
    obs, _ = env.reset()
    obs_t = _policy_obs(obs)
    total_reward = torch.zeros(1, device=device)
    print(f"[INFO] Loaded exported teacher: {args_cli.policy}", flush=True)
    print("[INFO] Stepping live exported teacher. Press Ctrl+C to stop.", flush=True)

    try:
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                actions = policy(obs_t.to(device))
                if args_cli.clip_actions > 0:
                    actions = actions.clamp(-args_cli.clip_actions, args_cli.clip_actions)
            obs, rewards, terms, truncs, _infos = env.step(actions.to(env.unwrapped.device))
            env.unwrapped.sim.render()
            obs_t = _policy_obs(obs)
            rewards = rewards.to(device)
            dones = terms | truncs
            total_reward += rewards
            progress = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
            if step % args_cli.print_every == 0 or bool(dones.any()) or bool(progress.success.any()):
                print(
                    f"step={step:04d} reward={rewards.mean().item():.4f} "
                    f"total_reward={total_reward.mean().item():.4f} "
                    f"action_norm={actions.norm(dim=-1).mean().item():.4f} "
                    f"success={progress.success.float().mean().item():.3f} "
                    f"xyz={progress.xyz_distance.mean().item():.4f} "
                    f"done={bool(dones.any())}",
                    flush=True,
                )
            if bool(dones.any()):
                obs, _ = env.reset()
                obs_t = _policy_obs(obs)
                total_reward.zero_()
            if args_cli.sleep > 0:
                time.sleep(args_cli.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
