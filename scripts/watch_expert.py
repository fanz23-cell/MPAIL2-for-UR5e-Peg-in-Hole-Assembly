#!/usr/bin/env python3
"""Watch expert policy in Isaac Sim.

Usage:
    conda run -n mpail2-omnireset python scripts/watch_expert.py \
        --checkpoint mpail2/train/isaac_franka/demos/peg_expert_seed42.pt \
        --output /tmp/expert_demo.mp4
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("/tmp/expert_demo.mp4"))
parser.add_argument("--num-steps", type=int, default=320)
parser.add_argument("--print-every", type=int, default=10)
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--no-record", action="store_true", help="Step in the live viewer without saving video frames.")
parser.add_argument("--clip-actions", type=float, default=1.0, help="Clip expert actions before stepping; <=0 disables.")
parser.add_argument(
    "--reset-types",
    nargs="+",
    default=None,
    help="Optional OmniReset reset_types override, e.g. ObjectAnywhereEEGrasped",
)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import torch, gymnasium as gym, imageio
import torch.nn as nn
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks, uwlab_tasks  # noqa

class _ObsNormalizer(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.register_buffer("_mean", state_dict["actor_obs_normalizer._mean"])
        self.register_buffer("_std",  state_dict["actor_obs_normalizer._std"])
    def forward(self, x):
        return (x - self._mean) / (self._std + 1e-8)

class _Actor(nn.Module):
    def __init__(self, state_dict, device):
        super().__init__()
        self.norm = _ObsNormalizer(state_dict)
        weight_keys = sorted([k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")],
                             key=lambda k: int(k.split(".")[1]))
        layers = []
        for i, wk in enumerate(weight_keys):
            o, inp = state_dict[wk].shape
            layers.append(nn.Linear(inp, o))
            if i < len(weight_keys) - 1:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        self.actor.load_state_dict({k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")})
        self.to(device)

    @torch.inference_mode()
    def forward(self, obs):
        return self.actor(self.norm(obs))

def main():
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    if args_cli.reset_types:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested --reset-types, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = list(args_cli.reset_types)
        params["probs"] = [1.0 / len(args_cli.reset_types)] * len(args_cli.reset_types)
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}")
    render_mode = None if args_cli.no_record else "rgb_array"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    ckpt = torch.load(str(args_cli.checkpoint), map_location=device, weights_only=False)
    policy = _Actor(ckpt["model_state_dict"], device)
    policy.eval()
    print(f"[INFO] Loaded expert checkpoint iter={ckpt.get('iter','?')}")

    frames = []
    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    total_reward = torch.zeros(1, device=device)

    for step in range(args_cli.num_steps):
        actions = policy(obs_t)
        if args_cli.clip_actions and args_cli.clip_actions > 0:
            actions = actions.clamp(-args_cli.clip_actions, args_cli.clip_actions)
        obs, rewards, terms, truncs, _ = env.step(actions)
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        rewards = rewards.to(device)
        total_reward += rewards

        if not args_cli.no_record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        dones = terms | truncs
        if step % args_cli.print_every == 0 or dones.any():
            print(
                f"step={step:04d} reward={rewards.mean().item():.4f} "
                f"total_reward={total_reward.mean().item():.4f} "
                f"action_norm={actions.norm(dim=-1).mean().item():.4f} done={bool(dones.any())}",
                flush=True,
            )

        if dones.any():
            print(f"  Episode ended at step {step}")
            obs, _ = env.reset()
            obs_t = obs["policy"] if isinstance(obs, dict) else obs
            total_reward.zero_()

        if args_cli.sleep > 0:
            time.sleep(args_cli.sleep)

    env.close()
    if args_cli.no_record:
        return
    if frames:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(args_cli.output), frames, fps=10)
        print(f"Video saved → {args_cli.output}  ({len(frames)} frames)")
    else:
        print("[WARN] No frames captured")

if __name__ == "__main__":
    main()
    simulation_app.close()
