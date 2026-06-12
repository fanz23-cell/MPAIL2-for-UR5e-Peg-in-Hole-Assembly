#!/usr/bin/env python3
"""Collect state-based demo trajectories from a pre-trained OmniReset expert checkpoint.

Rolls out the RSL-RL checkpoint in Isaac Sim and saves (obs, next_obs) pairs
as a .pt file that scripts/convert_omnireset_demo.py can ingest.

Bypasses OnPolicyRunner entirely to avoid rsl_rl version-API mismatches:
loads the actor MLP and obs-normalizer directly from model_state_dict.

Usage:
    conda run -n mpail2-omnireset python scripts/collect_state_demos_omnireset.py \
        --checkpoint mpail2/train/isaac_franka/demos/peg_expert_seed42.pt \
        --output     mpail2/train/isaac_franka/demos/peg_state_raw.pt \
        --num-envs 32 --num-steps 5000 --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect OmniReset peg-in-hole state demos from an expert checkpoint.")
parser.add_argument("--checkpoint", type=Path, required=True, help="Path to finetuned state expert .pt file")
parser.add_argument("--output", type=Path, required=True, help="Output raw demo .pt path")
parser.add_argument("--num-envs", type=int, default=32, help="Parallel Isaac Sim environments")
parser.add_argument("--num-steps", type=int, default=5000, help="Total env-steps to collect across all envs")
parser.add_argument(
    "--task",
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0",
    help="Registered Isaac Sim task ID to run",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--reset-types",
    nargs="+",
    default=None,
    help="Optional OmniReset reset_types override, e.g. ObjectPartiallyAssembledEEGrasped",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Clear leftover argv so downstream OmniKit parsers see nothing unexpected.
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything below runs after Isaac Sim is live ─────────────────────────

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402


# ── minimal actor that mirrors the checkpoint layout ──────────────────────

class _ObsNormalizer(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, obs_dim))
        self.register_buffer("_std", torch.ones(1, obs_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 1e-8)


class _ActorPolicy(nn.Module):
    """Reconstruct actor MLP from checkpoint state_dict shapes; no rsl_rl dependency."""

    def __init__(self, state_dict: dict, device: str):
        super().__init__()

        # obs normalizer
        obs_dim = state_dict["actor_obs_normalizer._mean"].shape[-1]
        self.normalizer = _ObsNormalizer(obs_dim)
        self.normalizer._mean.copy_(state_dict["actor_obs_normalizer._mean"])
        self.normalizer._std.copy_(state_dict["actor_obs_normalizer._std"])

        # infer layer sizes from weight-tensor shapes, sorted by layer index
        weight_keys = sorted(
            [k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        layers: list[nn.Module] = []
        for i, wk in enumerate(weight_keys):
            out_dim, in_dim = state_dict[wk].shape
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(weight_keys) - 1:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)

        # load actor weights
        actor_sd = {k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")}
        self.actor.load_state_dict(actor_sd)
        self.to(device)

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    device: str = args_cli.device or "cuda"

    # build env
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    if args_cli.reset_types:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested --reset-types, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = list(args_cli.reset_types)
        params["probs"] = [1.0 / len(args_cli.reset_types)] * len(args_cli.reset_types)
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}")
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # load checkpoint directly — bypasses OnPolicyRunner / rsl_rl version issues
    ckpt = torch.load(str(args_cli.checkpoint), map_location=device, weights_only=False)
    policy = _ActorPolicy(ckpt["model_state_dict"], device=device)
    policy.eval()
    print(f"[INFO] Loaded actor from checkpoint (iter {ckpt.get('iter', '?')})")

    # rollout
    def _extract_policy_obs(o) -> torch.Tensor:
        """env.get_observations() may return a TensorDict; pull out the 'policy' group."""
        if isinstance(o, torch.Tensor):
            return o
        # TensorDict / dict-like
        return o["policy"]

    raw_obs = env.get_observations()
    obs = _extract_policy_obs(raw_obs)
    obs_list: list[torch.Tensor] = []
    action_list: list[torch.Tensor] = []
    next_obs_list: list[torch.Tensor] = []

    target = args_cli.num_steps
    collected = 0
    print(f"Collecting {target} steps across {args_cli.num_envs} envs ...")

    while collected < target and simulation_app.is_running():
        actions = policy(obs)
        raw_next_obs, _, _, _ = env.step(actions)
        next_obs = _extract_policy_obs(raw_next_obs)

        obs_list.append(obs.cpu())
        action_list.append(actions.cpu())
        next_obs_list.append(next_obs.cpu())
        obs = next_obs
        collected += args_cli.num_envs

        if collected % (args_cli.num_envs * 100) == 0:
            print(f"  {collected}/{target} steps")

    env.close()

    # save in format that convert_omnireset_demo.py expects
    obs_tensor = torch.cat(obs_list, dim=0)
    actions_tensor = torch.cat(action_list, dim=0)
    nobs_tensor = torch.cat(next_obs_list, dim=0)

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data": {
                "obs": {"policy": obs_tensor},
                "actions": actions_tensor,
                "next_obs": {"policy": nobs_tensor},
            }
        },
        args_cli.output,
    )
    print(f"Saved {len(obs_tensor)} transitions -> {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
