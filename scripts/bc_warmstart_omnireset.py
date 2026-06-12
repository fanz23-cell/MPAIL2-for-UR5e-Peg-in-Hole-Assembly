#!/usr/bin/env python3
"""Behavior-cloning warm start for MPAIL2 OmniReset state policy.

The input is a raw rollout file from ``collect_state_demos_omnireset.py`` that
contains ``data.obs.policy``, ``data.actions``, and ``data.next_obs.policy``.
The output is a lightweight MPAIL checkpoint containing a warm-started planner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mpail2.learner import MPAIL2Learner
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig


def _load_raw(path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    obs = data["data"]["obs"]["policy"].float()
    next_obs = data["data"]["next_obs"]["policy"].float()
    actions = data["data"]["actions"].float()
    demos = {"policy": torch.stack([obs, next_obs], dim=1)}
    return demos, actions


def _policy_first_action_mean(policy, z: torch.Tensor) -> torch.Tensor:
    out = policy.model(z)
    mu, _log_std = torch.chunk(out, 2, dim=-1)
    mu = mu.view(z.shape[0], policy.horizon, policy.action_dim)
    return torch.tanh(mu[:, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description="BC warm-start OmniReset MPAIL policy.")
    parser.add_argument("--input", type=Path, required=True, help="Raw demo .pt with obs/actions/next_obs")
    parser.add_argument("--output", type=Path, required=True, help="Output warm-start checkpoint")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--action-scale-quantile",
        type=float,
        default=1.0,
        help="Per-dim abs-action quantile used to normalize raw teacher actions into [-1, 1].",
    )
    args = parser.parse_args()

    demos_cpu, actions_cpu = _load_raw(args.input)
    action_scale = torch.quantile(actions_cpu.abs(), args.action_scale_quantile, dim=0).clamp_min(1.0)
    normalized_actions_cpu = (actions_cpu / action_scale).clamp(-1.0, 1.0)
    demos = {k: v.to(args.device) for k, v in demos_cpu.items()}
    actions = normalized_actions_cpu.to(args.device)
    action_scale = action_scale.to(args.device)
    print(f"[INFO] action_scale={action_scale.detach().cpu().tolist()}", flush=True)

    cfg = OmniResetStateLearnerConfig()
    learner = MPAIL2Learner(demos, num_envs=1, learner_config=cfg, device=args.device)
    learner.planner.train()

    obs = demos["policy"][:, 0]
    dataset = TensorDataset(obs, actions)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(
        list(learner._encoder.parameters()) + list(learner._policy.parameters()),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_action_norm = 0.0
        total = 0
        for obs_batch, action_batch in loader:
            obs_dict = {"policy": obs_batch}
            if learner._obs_normalizer is not None:
                obs_dict = learner._obs_normalizer(obs_dict)
            z = learner._encoder(obs_dict)
            pred = _policy_first_action_mean(learner._policy, z)
            loss = F.mse_loss(pred, action_batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(learner._encoder.parameters()) + list(learner._policy.parameters()),
                max_norm=10.0,
            )
            opt.step()

            batch = obs_batch.shape[0]
            total_loss += loss.item() * batch
            total_action_norm += pred.norm(dim=-1).mean().item() * batch
            total += batch

        print(
            f"[bc epoch {epoch + 1:03d}/{args.epochs}] "
            f"loss={total_loss / total:.6f} pred_action_norm={total_action_norm / total:.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": learner.planner.state_dict(),
            "iter": f"bc_{args.epochs}",
            "source_demo": str(args.input),
            "action_scale": action_scale.detach().cpu(),
            "action_scale_quantile": args.action_scale_quantile,
        },
        args.output,
    )
    print(f"Saved BC warm-start checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
