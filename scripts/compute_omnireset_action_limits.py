#!/usr/bin/env python3
"""Estimate OmniReset raw action limits from raw-action demonstration files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_actions(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        for key in ("actions", "action", "raw_actions"):
            if key in data:
                return torch.as_tensor(data[key], dtype=torch.float32).reshape(-1, 7)
        nested = data.get("data")
        if isinstance(nested, dict):
            for key in ("actions", "action", "raw_actions"):
                if key in nested:
                    return torch.as_tensor(nested[key], dtype=torch.float32).reshape(-1, 7)
    raise KeyError(f"Could not find raw actions in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo",
        nargs="?",
        default="mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt",
        help="Raw-action BC demo file.",
    )
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument("--pad", type=float, default=1.2)
    parser.add_argument("--floor", type=float, default=1.0)
    args = parser.parse_args()

    actions = _load_actions(Path(args.demo))
    abs_actions = actions.abs()
    q = torch.quantile(abs_actions, args.quantile, dim=0)
    limit = torch.clamp(q * args.pad, min=args.floor)

    print("count", actions.shape[0])
    print("mean", [round(float(x), 4) for x in actions.mean(dim=0)])
    print("std", [round(float(x), 4) for x in actions.std(dim=0)])
    print(f"q{args.quantile:g}", [round(float(x), 4) for x in q])
    print("hydra_limit", "[" + ",".join(f"{float(x):.4g}" for x in limit) + "]")


if __name__ == "__main__":
    main()
