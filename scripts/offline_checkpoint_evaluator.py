#!/usr/bin/env python3
"""Offline diagnostics for MPAIL2 checkpoints without Isaac rollout rendering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import mpail2.configs.cfgs as cfgs
from mpail2.learner import MPAIL2Learner
from mpail2.envs.isaac.omnireset_wrapper import (
    compact_omnireset_policy_obs,
    planner_v2_omnireset_policy_obs,
    planner_v3_omnireset_policy_obs,
)
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig


DEFAULT_RAW_ACTION_LIMIT = [8.0, 8.0, 8.0, 5.0, 5.0, 1.25, 0.5]


def _load_demo(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor | None, dict[str, Any]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}

    if isinstance(raw, dict) and "demonstrations" in raw:
        demos = raw["demonstrations"]
        obs = {k: v[:, 0].float() for k, v in demos.items()}
        next_obs = {k: v[:, 1].float() for k, v in demos.items()}
        return obs, next_obs, None, metadata

    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
        obs = {k: v.float() for k, v in data["obs"].items()}
        next_obs = {k: v.float() for k, v in data["next_obs"].items()}
        actions = data.get("actions")
        if actions is not None:
            actions = actions.float()
        return obs, next_obs, actions, metadata

    raise ValueError(f"Unsupported demo format in {path}")


def _infer_horizon(state_dict: dict[str, torch.Tensor], action_dim: int) -> int:
    for key, value in state_dict.items():
        if key.endswith("sampling.policy.model.6.weight"):
            return int(value.shape[0] // (2 * action_dim))
    raise KeyError("Could not infer policy horizon from checkpoint state_dict")


def _infer_obs_dim(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["encoder.coder.0.weight"].shape[1])


def _adapt_demo_obs(
    demo_obs: dict[str, torch.Tensor],
    demo_next_obs: dict[str, torch.Tensor],
    obs_dim: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    current_dim = demo_obs["policy"].shape[-1]
    if current_dim == obs_dim:
        return demo_obs, demo_next_obs

    if current_dim == 215 and obs_dim in (60, 119):
        transform = planner_v3_omnireset_policy_obs if obs_dim == 119 else planner_v2_omnireset_policy_obs
        return (
            {"policy": transform(demo_obs["policy"])},
            {"policy": transform(demo_next_obs["policy"])},
        )

    if current_dim == 215 and obs_dim in (13, 19, 22):
        include_ee_to_hole = obs_dim in (19, 22)
        include_ee_vel = obs_dim == 22
        return (
            {"policy": compact_omnireset_policy_obs(
                demo_obs["policy"],
                include_ee_to_hole=include_ee_to_hole,
                include_ee_vel=include_ee_vel,
            )},
            {"policy": compact_omnireset_policy_obs(
                demo_next_obs["policy"],
                include_ee_to_hole=include_ee_to_hole,
                include_ee_vel=include_ee_vel,
            )},
        )

    raise ValueError(f"Demo obs dim {current_dim} cannot be adapted to checkpoint obs dim {obs_dim}")


def _build_learner(
    demo_obs: dict[str, torch.Tensor],
    demo_next_obs: dict[str, torch.Tensor],
    ckpt: dict[str, Any],
    device: str,
    std_floor: float,
    skip_low_std_dims: bool,
    reward_center: bool,
    reward_clip: float | None,
    obs_dim: int,
    latent_dim: int,
    identity_encoder: bool,
    reward_use_action: bool,
    residual_dynamics: bool,
) -> MPAIL2Learner:
    state_dict = ckpt["model_state_dict"]
    action_dim = 7
    horizon = _infer_horizon(state_dict, action_dim)
    demos = {k: torch.stack([demo_obs[k], demo_next_obs[k]], dim=1).to(device) for k in demo_obs}

    cfg = OmniResetStateLearnerConfig()
    cfg.planner_cfg.obs_dim = obs_dim
    cfg.planner_cfg.latent_dim = latent_dim
    if identity_encoder:
        cfg.planner_cfg.encoder_cfg = cfgs.IdentityCoderCfg(
            obs_key="policy",
            input_dim=obs_dim,
            output_dim=latent_dim,
        )
    else:
        cfg.planner_cfg.encoder_cfg.input_dim = obs_dim
        cfg.planner_cfg.encoder_cfg.output_dim = latent_dim
    cfg.planner_cfg.sampling_cfg.num_timesteps = horizon
    cfg.loss_horizon = horizon
    cfg.obs_normalizer_cfg.std_floor = std_floor
    cfg.obs_normalizer_cfg.skip_low_std_dims = skip_low_std_dims
    cfg.reward_center_for_planning = reward_center
    cfg.reward_clip_for_planning = reward_clip
    cfg.planner_cfg.reward_center_for_planning = reward_center
    cfg.planner_cfg.reward_clip_for_planning = reward_clip
    cfg.planner_cfg.reward_cfg.use_action = reward_use_action
    cfg.planner_cfg.dynamics_cfg.residual = residual_dynamics
    if cfg.planner_cfg.dynamics_cfg.model_kwargs is not None:
        cfg.planner_cfg.dynamics_cfg.model_kwargs["override_last_layer_norm"] = any(
            k.startswith("dynamics.model.7.") for k in state_dict
        )
    cfg.planner_cfg.__post_init__()

    for model_cfg in (
        cfg.planner_cfg.encoder_cfg,
        cfg.planner_cfg.reward_cfg,
        cfg.planner_cfg.value_cfg,
        cfg.planner_cfg.sampling_cfg,
        cfg.planner_cfg.dynamics_cfg,
    ):
        if getattr(model_cfg, "model_kwargs", None) is not None:
            model_cfg.model_kwargs["device"] = torch.device(device)

    learner = MPAIL2Learner(demos, num_envs=1, learner_config=cfg, device=device)
    learner.planner.load_state_dict(state_dict, strict=True)
    learner.planner.eval()
    return learner


def _policy_first_action_mean(policy, z: torch.Tensor) -> torch.Tensor:
    out = policy.model(z)
    mu, _log_std = torch.chunk(out, 2, dim=-1)
    mu = mu.view(z.shape[0], policy.horizon, policy.action_dim)
    return torch.tanh(mu[:, 0])


def _tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        f"{prefix}_mean": x.mean().item(),
        f"{prefix}_std": x.std().item() if x.numel() > 1 else 0.0,
        f"{prefix}_p10": x.quantile(0.10).item(),
        f"{prefix}_p50": x.quantile(0.50).item(),
        f"{prefix}_p90": x.quantile(0.90).item(),
        f"{prefix}_min": x.min().item(),
        f"{prefix}_max": x.max().item(),
    }


def _score_action(
    learner: MPAIL2Learner,
    z: torch.Tensor,
    next_z_actual: torch.Tensor,
    action: torch.Tensor,
    name: str,
    zero_next_z_dyn: torch.Tensor | None = None,
) -> dict[str, float]:
    next_z_dyn = learner._dynamics(z0=z, controls=action.unsqueeze(1))[:, 1]
    dyn_delta = next_z_dyn - z
    actual_delta = next_z_actual - z
    r_dyn = learner._calibrate_rewards(learner._reward(z, next_z_dyn, action=action))
    q_all = learner._value(z, action, return_type="all")
    q_mean_per_sample = q_all.mean(0)
    q_std_per_sample = q_all.std(0)
    actual_r = learner._calibrate_rewards(learner._reward(z, next_z_actual, action=action))

    out = {}
    out.update(_tensor_stats(f"{name}_reward_dyn", r_dyn))
    out.update(_tensor_stats(f"{name}_reward_actual_next", actual_r))
    out.update(_tensor_stats(f"{name}_q_mean", q_mean_per_sample))
    out.update(_tensor_stats(f"{name}_q_ensemble_std", q_std_per_sample))
    out.update(_tensor_stats(f"{name}_dyn_delta_norm", dyn_delta.norm(dim=-1)))
    out.update(_tensor_stats(f"{name}_dyn_demo_delta_cos", F.cosine_similarity(dyn_delta, actual_delta, dim=-1)))
    if zero_next_z_dyn is not None:
        out.update(_tensor_stats(f"{name}_dyn_vs_zero_action_norm", (next_z_dyn - zero_next_z_dyn).norm(dim=-1)))
    return out


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in row.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        flat[key] = value
    return flat


def evaluate_checkpoint(
    checkpoint: Path,
    demo: Path,
    args: argparse.Namespace,
    demo_obs: dict[str, torch.Tensor],
    demo_next_obs: dict[str, torch.Tensor],
    demo_actions: torch.Tensor | None,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=args.device, weights_only=False)
    obs_dim = args.obs_dim if args.obs_dim is not None else _infer_obs_dim(ckpt["model_state_dict"])
    latent_dim = args.latent_dim if args.latent_dim is not None else 512
    demo_obs, demo_next_obs = _adapt_demo_obs(demo_obs, demo_next_obs, obs_dim)
    learner = _build_learner(
        demo_obs,
        demo_next_obs,
        ckpt,
        args.device,
        args.std_floor,
        args.skip_low_std_dims,
        args.reward_center_for_planning,
        args.reward_clip_for_planning,
        obs_dim,
        latent_dim,
        args.identity_encoder,
        args.reward_use_action,
        args.residual_dynamics,
    )

    num_samples = demo_obs["policy"].shape[0]
    if args.sample_count < num_samples:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        indices = torch.randperm(num_samples, generator=generator)[: args.sample_count]
    else:
        indices = torch.arange(num_samples)

    obs = {k: v[indices].to(args.device) for k, v in demo_obs.items()}
    next_obs = {k: v[indices].to(args.device) for k, v in demo_next_obs.items()}
    if learner._obs_normalizer is not None:
        obs = learner._obs_normalizer(obs)
        next_obs = learner._obs_normalizer(next_obs)

    with torch.inference_mode():
        z = learner._encoder(obs)
        next_z = learner._encoder(next_obs)
        policy_action = _policy_first_action_mean(learner._policy, z)
        zero_action = torch.zeros_like(policy_action)
        random_action = torch.empty_like(policy_action).uniform_(-1.0, 1.0)
        zero_next_z_dyn = learner._dynamics(z0=z, controls=zero_action.unsqueeze(1))[:, 1]

        row: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "demo": str(demo),
            "checkpoint_iter": ckpt.get("iter"),
            "num_demo_transitions": num_samples,
            "sample_count": int(indices.numel()),
            "horizon": learner._policy.horizon,
            "reward_center_buffer": learner.planner._reward_center.item(),
        }

        raw_limit = torch.as_tensor(args.raw_action_limit, device=args.device, dtype=torch.float32).view(1, -1)
        for name, action in {
            "policy": policy_action,
            "zero": zero_action,
            "random": random_action,
        }.items():
            raw_action = action * raw_limit
            row.update(_tensor_stats(f"{name}_action_norm", action.norm(dim=-1)))
            row.update(_tensor_stats(f"{name}_raw_action_norm", raw_action.norm(dim=-1)))
            row[f"{name}_clip_frac_abs_gt_0p98"] = (action.abs() > 0.98).float().mean().item()
            for dim in range(action.shape[-1]):
                row[f"{name}_action_dim{dim}_mean"] = action[:, dim].mean().item()
                row[f"{name}_action_dim{dim}_std"] = action[:, dim].std().item()
            row.update(_score_action(learner, z, next_z, action, name, zero_next_z_dyn=zero_next_z_dyn))

        row["q_policy_minus_zero"] = row["policy_q_mean_mean"] - row["zero_q_mean_mean"]
        row["q_random_minus_zero"] = row["random_q_mean_mean"] - row["zero_q_mean_mean"]
        row["r_policy_dyn_minus_zero"] = row["policy_reward_dyn_mean"] - row["zero_reward_dyn_mean"]
        row["r_random_dyn_minus_zero"] = row["random_reward_dyn_mean"] - row["zero_reward_dyn_mean"]

        if demo_actions is not None:
            teacher_raw = demo_actions[indices].to(args.device)
            teacher_norm = (teacher_raw / raw_limit).clamp(-1.0, 1.0)
            row.update(_tensor_stats("teacher_action_norm", teacher_norm.norm(dim=-1)))
            row.update(_tensor_stats("teacher_raw_action_norm", teacher_raw.norm(dim=-1)))
            row.update(_score_action(learner, z, next_z, teacher_norm, "teacher", zero_next_z_dyn=zero_next_z_dyn))
            row["policy_teacher_norm_l2"] = (policy_action - teacher_norm).norm(dim=-1).mean().item()
            row["policy_teacher_raw_l2"] = ((policy_action * raw_limit) - teacher_raw).norm(dim=-1).mean().item()
            row["policy_teacher_raw_cos"] = F.cosine_similarity(policy_action * raw_limit, teacher_raw, dim=-1).mean().item()

    return _flatten_row(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std-floor", type=float, default=0.03)
    parser.add_argument("--skip-low-std-dims", action="store_true")
    parser.add_argument("--reward-center-for-planning", action="store_true")
    parser.add_argument("--reward-clip-for-planning", type=float, default=5.0)
    parser.add_argument("--raw-action-limit", type=float, nargs=7, default=DEFAULT_RAW_ACTION_LIMIT)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--identity-encoder", action="store_true")
    parser.add_argument("--reward-use-action", action="store_true")
    parser.add_argument("--residual-dynamics", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    demo_obs, demo_next_obs, demo_actions, _metadata = _load_demo(args.demo)
    rows = [
        evaluate_checkpoint(path, args.demo, args, demo_obs, demo_next_obs, demo_actions)
        for path in args.checkpoint
    ]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".json":
            args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        else:
            with args.output.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
                writer.writeheader()
                writer.writerows(rows)

    for row in rows:
        print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
