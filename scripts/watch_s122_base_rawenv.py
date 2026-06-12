#!/usr/bin/env python3
"""Watch S122 base policy on the raw Isaac Lab OmniReset env, without wrappers."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Direct raw-env viewer for S122 base policy.")
parser.add_argument("--checkpoint", type=Path, default=Path("logs/obs22_S122train_S115_rawObsTeacherBC_DAgger32_continue100_20260531_131911/models/model_100.pt"))
parser.add_argument("--demo-path", type=Path, default=Path("mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt"))
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--print-every", type=int, default=5)
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--reset-type", default="ObjectAnywhereEEAnywhere")
parser.add_argument("--raw-action-limit", nargs=7, type=float, default=[40.0, 40.0, 80.0, 30.0, 80.0, 15.0, 35.0])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402


def _prefixed_state(state: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _make_policy() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(215, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Linear(512, 42),
    )


def _load_demo_normalizer(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    demo = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(demo, dict) and "demonstrations" in demo:
        raw = demo["demonstrations"]["policy"].reshape(-1, 215).float()
    elif isinstance(demo, dict) and "data" in demo:
        raw = torch.cat(
            [demo["data"]["obs"]["policy"].float(), demo["data"]["next_obs"]["policy"].float()],
            dim=0,
        ).reshape(-1, 215)
    else:
        raise RuntimeError(f"Unsupported demo format: {path}")
    return raw.mean(dim=0).to(device), torch.clamp(raw.std(dim=0), min=0.03).to(device)


def _load_actor(checkpoint: Path, demo_path: Path, device: str):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    policy = _make_policy().to(device).eval()
    policy.load_state_dict(_prefixed_state(state, "sampling.policy.model."))
    mean, std = _load_demo_normalizer(demo_path, device)
    print(f"[INFO] Loaded S122 actor iter={ckpt.get('iter', '?')}", flush=True)
    return policy, mean, std


def _act(obs: dict[str, torch.Tensor], policy, mean, std, raw_limit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = obs["policy"].to(device=raw_limit.device, dtype=torch.float32)
    x = torch.clamp((x - mean) / (std + 1e-8), -10.0, 10.0)
    out = policy(x)
    mu, _ = torch.chunk(out, 2, dim=-1)
    norm_action = torch.tanh(mu.view(mu.shape[0], 3, 7)[:, 0])
    raw_action = norm_action * raw_limit
    return norm_action, raw_action


def main() -> None:
    device = args_cli.device or "cuda"
    policy, mean, std = _load_actor(args_cli.checkpoint, args_cli.demo_path, device)
    raw_limit = torch.tensor(args_cli.raw_action_limit, device=device, dtype=torch.float32).view(1, 7)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env_cfg.seed = args_cli.seed
    if args_cli.reset_type:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is not None:
            params["reset_types"] = [args_cli.reset_type]
            params["probs"] = [1.0]
            print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        obs, _ = env.reset()
        total_reward = torch.zeros(1, device=device)
        print("[INFO] Stepping S122 base on raw env. Press Ctrl+C to stop.", flush=True)
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                norm_action, raw_action = _act(obs, policy, mean, std, raw_limit)
            obs, rewards, terms, truncs, _ = env.step(raw_action.to(env.unwrapped.device))
            env.unwrapped.sim.render()
            rewards = rewards.to(device)
            dones = terms | truncs
            total_reward += rewards
            progress = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
            if step % args_cli.print_every == 0 or bool(dones.any()):
                print(
                    f"step={step:04d} reward={rewards.mean().item():.4f} "
                    f"total_reward={total_reward.mean().item():.4f} "
                    f"norm_action={norm_action.norm(dim=-1).mean().item():.4f} "
                    f"raw_action={raw_action.norm(dim=-1).mean().item():.4f} "
                    f"success={progress.success.float().mean().item():.3f} "
                    f"xyz={progress.xyz_distance.mean().item():.4f} done={bool(dones.any())}",
                    flush=True,
                )
            if bool(dones.any()):
                env.unwrapped.cfg.seed = args_cli.seed
                obs, _ = env.reset()
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
