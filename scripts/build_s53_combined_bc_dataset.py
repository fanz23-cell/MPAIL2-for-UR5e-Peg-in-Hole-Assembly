#!/usr/bin/env python3
"""Merge raw_obs/raw_action BC datasets for the S53 bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    obs_parts = []
    next_obs_parts = []
    action_parts = []
    metas = []
    for path_str in args.inputs:
        path = Path(path_str)
        data = torch.load(path, map_location="cpu", weights_only=False)
        payload = data["data"] if isinstance(data, dict) and "data" in data else data
        obs = payload["obs"]["policy"].float()
        next_obs = payload.get("next_obs", {}).get("policy", obs).float()
        actions = payload["actions"].float()
        obs_parts.append(obs)
        next_obs_parts.append(next_obs)
        action_parts.append(actions)
        metas.append({"path": str(path), "num_samples": int(obs.shape[0])})

    merged = {
        "data": {
            "obs": {"policy": torch.cat(obs_parts, dim=0)},
            "next_obs": {"policy": torch.cat(next_obs_parts, dim=0)},
            "actions": torch.cat(action_parts, dim=0),
        },
        "meta": {
            "source": "s53_combined_bc",
            "actions": "raw_env_units",
            "inputs": metas,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)
    actions = merged["data"]["actions"]
    print(f"saved {out}")
    print(f"samples {merged['data']['obs']['policy'].shape[0]}")
    print(f"obs_shape {tuple(merged['data']['obs']['policy'].shape)}")
    print(f"action_abs_max {actions.abs().max(dim=0).values.tolist()}")
    print(f"action_std {actions.std(dim=0).tolist()}")


if __name__ == "__main__":
    main()
