#!/usr/bin/env python3
"""Standalone supervised check for OmniReset demo action causality.

This intentionally bypasses MPAIL/reward/value/MPPI. It answers one question:
can a simple model learn that demo teacher actions predict demo next compact
observations better than zero/random actions?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mpail2.envs.isaac.omnireset_wrapper import compact_omnireset_policy_obs


def mlp(in_dim: int, out_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 512),
        torch.nn.LayerNorm(512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 512),
        torch.nn.LayerNorm(512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, out_dim),
    )


def load_demo(path: Path, obs_dim: int, raw_limit: torch.Tensor, device: str):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    obs = data["obs"]["policy"].float()
    next_obs = data["next_obs"]["policy"].float()
    actions = data["actions"].float()
    if obs.shape[-1] == 215:
        kwargs = {
            "include_ee_to_hole": obs_dim in (19, 22),
            "include_ee_vel": obs_dim == 22,
        }
        obs = compact_omnireset_policy_obs(obs, **kwargs)
        next_obs = compact_omnireset_policy_obs(next_obs, **kwargs)
    actions = (actions / raw_limit.cpu().view(1, -1)).clamp(-1.0, 1.0)
    return obs.to(device), next_obs.to(device), actions.to(device)


@torch.no_grad()
def eval_model(model, obs, next_obs, actions):
    pred_teacher = model(torch.cat([obs, actions], dim=-1))
    pred_zero = model(torch.cat([obs, torch.zeros_like(actions)], dim=-1))
    pred_rand = model(torch.cat([obs, torch.empty_like(actions).uniform_(-1.0, 1.0)], dim=-1))
    teacher_dist = (pred_teacher - next_obs).pow(2).mean(dim=-1)
    zero_dist = (pred_zero - next_obs).pow(2).mean(dim=-1)
    rand_dist = (pred_rand - next_obs).pow(2).mean(dim=-1)
    return {
        "teacher_dist": teacher_dist.mean().item(),
        "zero_dist": zero_dist.mean().item(),
        "rand_dist": rand_dist.mean().item(),
        "margin_zero": (zero_dist - teacher_dist).mean().item(),
        "margin_rand": (rand_dist - teacher_dist).mean().item(),
        "teacher_win_zero_frac": (teacher_dist < zero_dist).float().mean().item(),
        "teacher_win_rand_frac": (teacher_dist < rand_dist).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, default=Path("mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt"))
    parser.add_argument("--obs-dim", type=int, default=22)
    parser.add_argument("--raw-action-limit", nargs=7, type=float, default=[40, 40, 80, 30, 80, 15, 35])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("logs/diagnose_demo_action_dynamics_supervised.json"))
    args = parser.parse_args()

    raw_limit = torch.tensor(args.raw_action_limit, dtype=torch.float32, device=args.device)
    obs, next_obs, actions = load_demo(args.demo, args.obs_dim, raw_limit, args.device)
    mean = obs.mean(0, keepdim=True)
    std = obs.std(0, keepdim=True).clamp_min(0.03)
    obs_n = (obs - mean) / std
    next_obs_n = (next_obs - mean) / std

    model = mlp(args.obs_dim + actions.shape[-1], args.obs_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n = obs.shape[0]
    last = {}
    for step in range(args.steps):
        idx = torch.randint(n, (min(args.batch_size, n),), device=args.device)
        pred = model(torch.cat([obs_n[idx], actions[idx]], dim=-1))
        loss = F.mse_loss(pred, next_obs_n[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in {0, 99, 999, args.steps - 1}:
            last = eval_model(model, obs_n, next_obs_n, actions)
            print(json.dumps({"step": step + 1, "loss": loss.item(), **last}, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(last, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
