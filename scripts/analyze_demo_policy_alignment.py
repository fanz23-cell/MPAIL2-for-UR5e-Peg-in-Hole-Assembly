#!/usr/bin/env python3
"""Offline alignment checks between saved expert demos, exported teacher, and base policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mpail2.layers import mlp_factory


RAW_LIMIT = torch.tensor([40.0, 40.0, 80.0, 30.0, 80.0, 15.0, 35.0])


def quantiles(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "p50": float(torch.quantile(x, 0.50)),
        "p90": float(torch.quantile(x, 0.90)),
        "p95": float(torch.quantile(x, 0.95)),
        "p99": float(torch.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def pair_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    diff = pred - target
    l2 = torch.linalg.vector_norm(diff, dim=-1)
    mae = diff.abs().mean(dim=-1)
    cos = F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
    return {
        "l2": quantiles(l2),
        "mae": quantiles(mae),
        "cosine": quantiles(cos),
        "per_dim_mae": [float(v) for v in diff.abs().mean(dim=0).detach().cpu()],
        "per_dim_rmse": [float(v) for v in torch.sqrt((diff * diff).mean(dim=0)).detach().cpu()],
    }


def build_policy_from_checkpoint(checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, int, int]:
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = saved["model_state_dict"] if isinstance(saved, dict) and "model_state_dict" in saved else saved
    prefix = "sampling.policy.model."
    model_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not model_state:
        raise RuntimeError(f"No sampling.policy.model.* keys found in {checkpoint_path}")

    input_dim = int(model_state["0.weight"].shape[1])
    hidden_dim = int(model_state["0.weight"].shape[0])
    output_dim = int(model_state["6.weight"].shape[0])
    action_dim = 7
    horizon = output_dim // (action_dim * 2)
    if output_dim != horizon * action_dim * 2:
        raise RuntimeError(f"Unexpected policy output dim {output_dim}")

    model = mlp_factory(
        input_dim=input_dim,
        hidden_dims=[hidden_dim, hidden_dim],
        activation="silu",
        output_dim=output_dim,
        use_layer_norm=True,
        device=torch.device(device),
    )
    model.load_state_dict(model_state)
    model.eval()
    return model, horizon, action_dim


def load_raw_normalizer(path: Path, obs_dim: int, device: str, std_floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    demo = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(demo, dict) and "demonstrations" in demo:
        raw_demo = demo["demonstrations"]["policy"].reshape(-1, obs_dim).float()
    elif isinstance(demo, dict) and "data" in demo:
        obs = demo["data"]["obs"]["policy"].float()
        next_obs = demo["data"]["next_obs"]["policy"].float()
        raw_demo = torch.cat([obs, next_obs], dim=0)
    else:
        raise ValueError(f"Unsupported normalizer demo format: {path}")
    mean = raw_demo.mean(dim=0).to(device)
    std = raw_demo.std(dim=0).to(device)
    if std_floor > 0:
        std = torch.clamp(std, min=std_floor)
    return mean, std


def mask_metrics(name: str, mask: torch.Tensor, base_norm: torch.Tensor, expert_norm: torch.Tensor) -> dict | None:
    n = int(mask.sum().item())
    if n == 0:
        return None
    return {"name": name, "n": n, "base_vs_saved_norm": pair_metrics(base_norm[mask], expert_norm[mask])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default="mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_success_rollouts_raw.pt")
    parser.add_argument("--teacher-policy", default="mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt")
    parser.add_argument("--base-checkpoint", default="logs/obs22_S122train_S115_rawObsTeacherBC_DAgger32_continue100_20260531_131911/models/model_100.pt")
    parser.add_argument("--normalizer-demo", default="mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt")
    parser.add_argument("--output", default="logs/S170_demo_policy_alignment.json")
    parser.add_argument("--top-csv", default="logs/S170_demo_policy_alignment_top_errors.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    device = args.device
    rollouts = torch.load(args.rollouts, map_location="cpu", weights_only=False)
    data = rollouts["data"]
    obs = data["obs"]["policy"].float().to(device)
    saved_raw = data["actions"].float().to(device)
    raw_limit = RAW_LIMIT.to(device)
    saved_norm = torch.clamp(saved_raw / raw_limit, -1.0, 1.0)

    teacher = torch.jit.load(args.teacher_policy, map_location=device).eval()
    base_model, horizon, action_dim = build_policy_from_checkpoint(Path(args.base_checkpoint), device)
    mean, std = load_raw_normalizer(Path(args.normalizer_demo), obs.shape[-1], device, std_floor=0.03)

    teacher_chunks = []
    base_norm_chunks = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], args.batch_size):
            batch = obs[start : start + args.batch_size]
            teacher_chunks.append(teacher(batch).float())
            batch_norm = torch.clamp((batch - mean) / (std + 1e-8), -10.0, 10.0)
            out = base_model(batch_norm)
            mu, _ = torch.chunk(out, 2, dim=-1)
            base_plan = torch.tanh(mu.view(mu.shape[0], horizon, action_dim))
            base_norm_chunks.append(base_plan[:, 0])
    teacher_raw = torch.cat(teacher_chunks, dim=0)
    base_norm = torch.cat(base_norm_chunks, dim=0)
    base_raw = base_norm * raw_limit
    teacher_norm = torch.clamp(teacher_raw / raw_limit, -1.0, 1.0)

    step = data.get("step_in_episode")
    episode = data.get("episode_id")
    success = data.get("success")
    xyz = data.get("xyz_distance")
    step = step.to(device) if torch.is_tensor(step) else torch.arange(obs.shape[0], device=device)
    episode = episode.to(device) if torch.is_tensor(episode) else torch.full((obs.shape[0],), -1, device=device)
    success = success.to(device).bool() if torch.is_tensor(success) else torch.zeros(obs.shape[0], dtype=torch.bool, device=device)
    xyz = xyz.to(device).float() if torch.is_tensor(xyz) else torch.linalg.vector_norm(obs[:, 24:27], dim=-1)

    bins = []
    bounds = [0.01, 0.03, 0.06, 0.10, 0.16, 0.24]
    prev = None
    for bound in bounds:
        mask = xyz < bound if prev is None else (xyz >= prev) & (xyz < bound)
        item = mask_metrics(f"xyz_{'lt' if prev is None else f'{prev:.2f}_to'}_{bound:.2f}", mask, base_norm, saved_norm)
        if item:
            bins.append(item)
        prev = bound
    item = mask_metrics("xyz_ge_0.24", xyz >= bounds[-1], base_norm, saved_norm)
    if item:
        bins.append(item)
    for name, mask in [
        ("step_early_lt40", step < 40),
        ("step_mid_40_100", (step >= 40) & (step < 100)),
        ("step_late_ge100", step >= 100),
        ("success_flag_false", ~success),
        ("success_flag_true", success),
    ]:
        item = mask_metrics(name, mask, base_norm, saved_norm)
        if item:
            bins.append(item)

    base_l2 = torch.linalg.vector_norm(base_norm - saved_norm, dim=-1)
    top_k = min(50, base_l2.numel())
    top_vals, top_idx = torch.topk(base_l2, k=top_k)
    with Path(args.top_csv).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "index", "episode_id", "step", "xyz_distance", "success", "base_saved_norm_l2"])
        for rank, (idx, val) in enumerate(zip(top_idx.detach().cpu().tolist(), top_vals.detach().cpu().tolist()), start=1):
            writer.writerow([
                rank,
                idx,
                int(episode[idx].item()),
                int(step[idx].item()),
                float(xyz[idx].item()),
                bool(success[idx].item()),
                float(val),
            ])

    result = {
        "inputs": {
            "rollouts": args.rollouts,
            "teacher_policy": args.teacher_policy,
            "base_checkpoint": args.base_checkpoint,
            "normalizer_demo": args.normalizer_demo,
            "device": device,
            "num_transitions": int(obs.shape[0]),
            "obs_dim": int(obs.shape[-1]),
            "base_horizon": int(horizon),
        },
        "teacher_vs_saved_raw": pair_metrics(teacher_raw, saved_raw),
        "teacher_vs_saved_norm": pair_metrics(teacher_norm, saved_norm),
        "base_vs_saved_raw": pair_metrics(base_raw, saved_raw),
        "base_vs_saved_norm": pair_metrics(base_norm, saved_norm),
        "base_vs_teacher_norm": pair_metrics(base_norm, teacher_norm),
        "stratified": bins,
        "top_errors_csv": args.top_csv,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
