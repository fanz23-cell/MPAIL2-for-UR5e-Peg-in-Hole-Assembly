#!/usr/bin/env python3
"""Offline diagnostics for IQ-Learn pretrain checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mpail2.learner import MPAIL2Learner
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig
from mpail2.train.utils.demos import load_demonstrations


def _round_list(t: torch.Tensor) -> list[float]:
    return [round(float(x), 4) for x in t.detach().cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demo-path", type=Path, required=True)
    parser.add_argument("--raw-demo-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--raw-action-limit", nargs=7, type=float, default=[40, 40, 80, 30, 80, 15, 35])
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = OmniResetStateLearnerConfig()
    cfg.planner_cfg.encoder_cfg.input_dim = 215
    cfg.planner_cfg.action_dim = 7
    cfg.demo_action_raw_limit = list(args.raw_action_limit)

    demos = load_demonstrations(str(args.demo_path), device=device)
    learner = MPAIL2Learner(demos, 1, cfg, device=device)
    checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    learner.planner.load_state_dict(checkpoint["model_state_dict"])
    learner.eval()
    learner._load_iql_demo_transitions(str(args.raw_demo_path))

    obs = {"policy": learner._iql_obs}
    norm_obs = learner._obs_normalizer(obs) if learner._obs_normalizer is not None else obs
    preds: list[torch.Tensor] = []
    acts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, learner._iql_obs.shape[0], args.batch_size):
            end = start + args.batch_size
            z = learner._encoder({"policy": norm_obs["policy"][start:end]})
            preds.append(learner._policy_mean_from_latent(z).detach())
            acts.append(learner._iql_actions[start:end].detach())

    pred = torch.cat(preds)
    act = torch.cat(acts)
    err = (pred - act).abs()
    print(f"checkpoint={args.checkpoint}")
    print(f"n={act.shape[0]}")
    print(f"global_mae={err.mean().item():.6f}")
    print(f"global_rmse={(pred - act).pow(2).mean().sqrt().item():.6f}")
    print(f"per_dim_mae={_round_list(err.mean(0))}")
    print(f"demo_abs_mean={_round_list(act.abs().mean(0))}")
    print(f"pred_abs_mean={_round_list(pred.abs().mean(0))}")
    print(f"demo_std={_round_list(act.std(0))}")
    print(f"pred_std={_round_list(pred.std(0))}")
    print(f"demo_abs_gt_0.95={(act.abs() > 0.95).float().mean().item():.6f}")
    print(f"pred_abs_gt_0.95={(pred.abs() > 0.95).float().mean().item():.6f}")
    print(f"not_done_mean={learner._iql_not_done.mean().item():.6f}")
    print(f"not_done_zeros={(learner._iql_not_done == 0).sum().item()}")


if __name__ == "__main__":
    main()
