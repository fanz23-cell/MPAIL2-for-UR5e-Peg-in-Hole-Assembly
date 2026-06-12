#!/usr/bin/env python3
"""Export an OmniReset RSL-RL state checkpoint to a TorchScript policy.

The official UWLab ``rsl_rl/play.py`` performs the same conceptual export, but
the vendored script can drift against the installed IsaacLab/RSL-RL API.  This
script reconstructs the deterministic actor and actor observation normalizer
directly from the checkpoint state dict and saves a self-contained
``policy.pt`` that maps policy observations to raw actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class ObsNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / (self.std + 1.0e-8)


class ActorPolicy(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.normalizer = ObsNormalizer(
            state_dict["actor_obs_normalizer._mean"].float(),
            state_dict["actor_obs_normalizer._std"].float(),
        )

        weight_keys = sorted(
            [key for key in state_dict if key.startswith("actor.") and key.endswith(".weight")],
            key=lambda key: int(key.split(".")[1]),
        )
        layers: list[nn.Module] = []
        for idx, weight_key in enumerate(weight_keys):
            out_dim, in_dim = state_dict[weight_key].shape
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(weight_keys) - 1:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        self.actor.load_state_dict({k[len("actor.") :]: v for k, v in state_dict.items() if k.startswith("actor.")})

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OmniReset state checkpoint to TorchScript policy.pt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = ActorPolicy(checkpoint["model_state_dict"]).eval()
    obs_dim = checkpoint["model_state_dict"]["actor_obs_normalizer._mean"].shape[-1]
    example_obs = torch.zeros(1, obs_dim)

    traced = torch.jit.trace(policy, example_obs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    print(f"Exported TorchScript policy -> {args.output}")
    print(f"checkpoint_iter={checkpoint.get('iter', '?')} obs_dim={obs_dim}")


if __name__ == "__main__":
    main()
