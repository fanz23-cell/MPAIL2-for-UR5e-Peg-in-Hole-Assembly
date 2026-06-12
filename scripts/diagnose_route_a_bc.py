#!/usr/bin/env python3
"""Diagnose BC warm-start action fit on an OmniReset route-A dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mpail2.learner import MPAIL2Learner
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig


def _policy_first_action_mean(policy, z: torch.Tensor) -> torch.Tensor:
    out = policy.model(z)
    mu, _log_std = torch.chunk(out, 2, dim=-1)
    mu = mu.view(z.shape[0], policy.horizon, policy.action_dim)
    return torch.tanh(mu[:, 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    raw = torch.load(args.demo, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    obs = raw["data"]["obs"]["policy"].float()
    next_obs = raw["data"]["next_obs"]["policy"].float()
    teacher_raw = raw["data"]["actions"].float()
    demos = {"policy": torch.stack([obs, next_obs], dim=1).to(args.device)}
    action_scale = ckpt["action_scale"].float().to(args.device)
    teacher_norm = (teacher_raw.to(args.device) / action_scale).clamp(-1.0, 1.0)

    learner = MPAIL2Learner(demos, num_envs=1, learner_config=OmniResetStateLearnerConfig(), device=args.device)
    learner.planner.load_state_dict(ckpt["model_state_dict"])
    learner.planner.eval()

    preds = []
    with torch.inference_mode():
        for start in range(0, obs.shape[0], args.batch_size):
            obs_batch = obs[start : start + args.batch_size].to(args.device)
            obs_dict = {"policy": obs_batch}
            if learner._obs_normalizer is not None:
                obs_dict = learner._obs_normalizer(obs_dict)
            z = learner._encoder(obs_dict)
            preds.append(_policy_first_action_mean(learner._policy, z))
    pred_norm = torch.cat(preds, dim=0)
    pred_raw = pred_norm * action_scale

    err_norm = pred_norm - teacher_norm
    err_raw = pred_raw - teacher_raw.to(args.device)
    cos = torch.nn.functional.cosine_similarity(pred_raw, teacher_raw.to(args.device), dim=-1)

    def stats(name: str, values: torch.Tensor) -> None:
        values = values.detach().float().cpu()
        print(
            f"{name}: mean={values.mean().item():.6f} "
            f"p50={values.quantile(0.50).item():.6f} "
            f"p90={values.quantile(0.90).item():.6f} "
            f"max={values.max().item():.6f}"
        )

    print(f"checkpoint_iter={ckpt.get('iter')}")
    print(f"num_samples={obs.shape[0]}")
    print(f"action_scale={action_scale.detach().cpu().tolist()}")
    stats("teacher_raw_norm", teacher_raw.norm(dim=-1))
    stats("pred_raw_norm", pred_raw.norm(dim=-1))
    stats("raw_l2_error", err_raw.norm(dim=-1))
    stats("norm_l2_error", err_norm.norm(dim=-1))
    stats("cosine_similarity", cos)
    print(f"teacher_raw_mean={teacher_raw.mean(dim=0).tolist()}")
    print(f"pred_raw_mean={pred_raw.detach().cpu().mean(dim=0).tolist()}")
    print(f"raw_mae_per_dim={err_raw.detach().abs().cpu().mean(dim=0).tolist()}")


if __name__ == "__main__":
    main()
