#!/usr/bin/env python3
"""Audit saved OmniReset successful rollout files episode by episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def longest_true_run(values: torch.Tensor) -> int:
    longest = 0
    current = 0
    for value in values.bool().tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--min-success-run", type=int, default=3)
    parser.add_argument("--max-min-xyz", type=float, default=0.01)
    args = parser.parse_args()

    payload = torch.load(str(args.input), map_location="cpu", weights_only=False)
    data = payload["data"]
    episode_id = data["episode_id"].long()
    success = data["success"].bool()
    xyz = data["xyz_distance"].float()
    action_norm = data["action_norm"].float()

    rows = []
    for episode in torch.unique(episode_id, sorted=True):
        idx = (episode_id == episode).nonzero(as_tuple=False).flatten()
        ep_success = success[idx]
        ep_xyz = xyz[idx]
        ep_action = action_norm[idx]
        first_success = int(ep_success.nonzero(as_tuple=False).flatten()[0]) if ep_success.any() else -1
        rows.append(
            {
                "episode": int(episode),
                "length": int(idx.numel()),
                "has_success": bool(ep_success.any()),
                "first_success": first_success,
                "success_count": int(ep_success.sum()),
                "longest_success_run": longest_true_run(ep_success),
                "min_xyz": float(ep_xyz.min()),
                "xyz_at_first_success": float(ep_xyz[first_success]) if first_success >= 0 else float("nan"),
                "max_action_norm": float(ep_action.max()),
            }
        )

    success_rows = [row for row in rows if row["has_success"]]
    strict_rows = [
        row
        for row in rows
        if row["has_success"]
        and row["longest_success_run"] >= args.min_success_run
        and row["min_xyz"] <= args.max_min_xyz
    ]
    rejected = [row for row in rows if row not in strict_rows]
    first_steps = torch.tensor([row["first_success"] for row in success_rows], dtype=torch.float32)
    min_xyz_values = torch.tensor([row["min_xyz"] for row in rows], dtype=torch.float32)

    print(f"input={args.input}")
    print(f"episodes={len(rows)}")
    print(f"episodes_with_any_success={len(success_rows)}")
    for run in [1, 2, 3, 5, 10, 20]:
        print(f"episodes_with_success_run>={run}={sum(row['longest_success_run'] >= run for row in rows)}")
    print(f"episodes_min_xyz<=0.005={sum(row['min_xyz'] <= 0.005 for row in rows)}")
    print(f"episodes_min_xyz<=0.010={sum(row['min_xyz'] <= 0.010 for row in rows)}")
    if len(first_steps) > 0:
        quantiles = torch.quantile(first_steps, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]))
        print(
            "first_success_step="
            f"mean:{first_steps.mean().item():.2f} "
            f"min:{first_steps.min().item():.0f} max:{first_steps.max().item():.0f} "
            f"q:{[round(float(x), 2) for x in quantiles.tolist()]}"
        )
    print(
        "min_xyz="
        f"mean:{min_xyz_values.mean().item():.5f} "
        f"min:{min_xyz_values.min().item():.5f} max:{min_xyz_values.max().item():.5f}"
    )
    print(
        f"strict_keep(min_success_run>={args.min_success_run}, min_xyz<={args.max_min_xyz})="
        f"{len(strict_rows)}"
    )
    print(f"strict_reject={len(rejected)}")
    print("first_rejected_rows=")
    for row in rejected[:20]:
        print(row)


if __name__ == "__main__":
    main()
