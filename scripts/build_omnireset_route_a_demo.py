#!/usr/bin/env python3
"""Slice route-A successful rollouts into BC and MPAIL demo files.

Route A keeps the whole "already grasped but not yet successful" prefix from
successful ObjectAnywhereEEGrasped episodes.  It intentionally does not filter
by peg-hole distance because this route is meant to learn carry + align +
insert, not only near-hole insertion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean route-A demos from successful rollout file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True, help="Raw obs/action/next_obs file for BC")
    parser.add_argument("--mpail-output", type=Path, required=True, help="MPAIL transition demo file")
    parser.add_argument("--min-action-norm", type=float, default=0.05)
    parser.add_argument("--min-pre-success-steps", type=int, default=5)
    args = parser.parse_args()

    src = torch.load(args.input, map_location="cpu", weights_only=False)
    data = src["data"]
    obs = data["obs"]["policy"].float()
    actions = data["actions"].float()
    next_obs = data["next_obs"]["policy"].float()
    success = data["success"].bool()
    episode_id = data["episode_id"].long()
    action_norm = data["action_norm"].float()
    xyz_distance = data["xyz_distance"].float()

    keep = torch.zeros(len(obs), dtype=torch.bool)
    episodes_used = 0
    episodes_skipped_initial_success = 0
    episodes_skipped_no_prefix = 0

    for ep in torch.unique(episode_id):
        idx = (episode_id == ep).nonzero(as_tuple=False).flatten()
        ep_success = success[idx]
        if not ep_success.any():
            continue
        first_success_local = int(ep_success.nonzero(as_tuple=False).flatten()[0])
        if first_success_local < args.min_pre_success_steps:
            episodes_skipped_initial_success += 1
            continue
        prefix_idx = idx[:first_success_local]
        prefix_keep = action_norm[prefix_idx] >= args.min_action_norm
        if not prefix_keep.any():
            episodes_skipped_no_prefix += 1
            continue
        keep[prefix_idx[prefix_keep]] = True
        episodes_used += 1

    kept_count = int(keep.sum())
    if kept_count == 0:
        raise RuntimeError("Route-A slicing produced zero transitions; refusing to save empty demos.")

    raw = {
        "data": {
            "obs": {"policy": obs[keep]},
            "actions": actions[keep],
            "next_obs": {"policy": next_obs[keep]},
            "action_norm": action_norm[keep],
            "xyz_distance": xyz_distance[keep],
            "episode_id": episode_id[keep],
        },
        "metadata": {
            **src.get("metadata", {}),
            "slice": "route_a_pre_success_object_anywhere_ee_grasped",
            "min_action_norm": args.min_action_norm,
            "min_pre_success_steps": args.min_pre_success_steps,
            "episodes_used": episodes_used,
            "kept_transitions": kept_count,
        },
    }
    mpail = {
        "demonstrations": {"policy": torch.stack([obs[keep], next_obs[keep]], dim=1)},
        "metadata": {
            **raw["metadata"],
            "format": "mpail_transition_dict",
            "obs_key": "policy",
            "source": str(args.input),
        },
    }

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.mpail_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, str(args.raw_output))
    torch.save(mpail, str(args.mpail_output))

    print(f"Input transitions          : {len(obs)}")
    print(f"Episodes used              : {episodes_used}")
    print(f"Skipped initial-success eps : {episodes_skipped_initial_success}")
    print(f"Skipped no-prefix eps       : {episodes_skipped_no_prefix}")
    print(f"Kept transitions           : {kept_count}")
    print(f"Action norm kept range      : [{action_norm[keep].min().item():.4f}, {action_norm[keep].max().item():.4f}]")
    print(f"XYZ distance kept range     : [{xyz_distance[keep].min().item():.4f}, {xyz_distance[keep].max().item():.4f}]")
    print(f"Saved raw BC demo           : {args.raw_output}")
    print(f"Saved MPAIL demo            : {args.mpail_output}")


if __name__ == "__main__":
    main()
