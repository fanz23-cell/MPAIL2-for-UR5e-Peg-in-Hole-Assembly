#!/usr/bin/env python3
"""Direct Isaac Lab viewer for the S122 raw-observation base policy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("logs/obs22_S122train_S115_rawObsTeacherBC_DAgger32_continue100_20260531_131911/models/model_100.pt"))
parser.add_argument("--demo-path", type=Path, default=Path("mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt"))
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--print-every", type=int, default=5)
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--reset-types", nargs="+", default=["ObjectAnywhereEEAnywhere"])
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

from mpail2.envs.isaac.omnireset_wrapper import OmniResetMPAILWrapper  # noqa: E402


def _prefixed_state(state: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _make_encoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(215, 256),
        nn.LayerNorm(256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.SiLU(),
        nn.Linear(256, 512),
        nn.LayerNorm(512),
    )


def _make_policy() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Linear(512, 98),
    )


def _load_demo_normalizer(path: Path, device: str):
    demo = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(demo, dict) and "demonstrations" in demo:
        obs = demo["demonstrations"]["policy"].reshape(-1, 215).float()
    elif isinstance(demo, dict) and "data" in demo:
        obs0 = demo["data"]["obs"]["policy"].float()
        obs1 = demo["data"]["next_obs"]["policy"].float()
        obs = torch.cat([obs0, obs1], dim=0).reshape(-1, 215)
    else:
        raise RuntimeError(f"Unsupported demo format: {path}")
    mean = obs.mean(dim=0).to(device)
    std = torch.clamp(obs.std(dim=0), min=0.03).to(device)
    return mean, std


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    if args_cli.reset_types:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        params["reset_types"] = list(args_cli.reset_types)
        params["probs"] = [1.0 / len(args_cli.reset_types)] * len(args_cli.reset_types)
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = OmniResetMPAILWrapper(
        env,
        raw_action_limit=args_cli.raw_action_limit,
        compact_obs_dim=215,
    )

    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    encoder = _make_encoder().to(device).eval()
    policy = _make_policy().to(device).eval()
    encoder.load_state_dict(_prefixed_state(state, "encoder.coder."))
    policy.load_state_dict(_prefixed_state(state, "sampling.policy.model."))
    mean, std = _load_demo_normalizer(args_cli.demo_path, device)
    print(f"[INFO] Loaded direct S122 base checkpoint iter={ckpt.get('iter', '?')}", flush=True)
    print("[INFO] Stepping direct encoder+policy mean. Press Ctrl+C to stop.", flush=True)

    obs, _ = env.reset()
    total_reward = torch.zeros(args_cli.num_envs, device=device)
    try:
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                x = obs["policy"].to(device)
                x = torch.clamp((x - mean) / (std + 1e-8), -10.0, 10.0)
                z = encoder(x)
                out = policy(z)
                mu, _ = torch.chunk(out, 2, dim=-1)
                actions = torch.tanh(mu.view(mu.shape[0], 7, 7)[:, 0])
            obs, rewards, terms, truncs, _ = env.step(actions)
            dones = terms | truncs
            rewards = rewards.to(device)
            total_reward += rewards
            progress = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
            if step % args_cli.print_every == 0 or bool(dones.any()):
                print(
                    f"step={step:04d} reward={rewards.mean().item():.4f} "
                    f"total_reward={total_reward.mean().item():.4f} "
                    f"action_norm={actions.norm(dim=-1).mean().item():.4f} "
                    f"success={progress.success.float().mean().item():.3f} "
                    f"xyz={progress.xyz_distance.mean().item():.4f} done={bool(dones.any())}",
                    flush=True,
                )
            if bool(dones.any()):
                env.unwrapped.cfg.seed = 42
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
