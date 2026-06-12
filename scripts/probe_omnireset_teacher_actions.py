#!/usr/bin/env python3
"""Probe whether expert actions causally improve OmniReset in the real env.

This is a training-free diagnostic.  It compares the official teacher policy,
the same teacher clipped through the MPAIL raw-action limits, zero actions, and
random limit-scaled actions under the same reset distribution.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="OmniReset real-env teacher action causal probe.")
parser.add_argument(
    "--policy",
    type=Path,
    default=Path("mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt"),
)
parser.add_argument("--output", type=Path, default=Path("logs/teacher_action_probe.csv"))
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--num-episodes", type=int, default=32)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument(
    "--reset-types",
    nargs="+",
    default=["ObjectAnywhereEEAnywhere"],
    help="OmniReset reset types to evaluate.",
)
parser.add_argument(
    "--raw-action-limit",
    nargs=7,
    type=float,
    default=[8.0, 8.0, 8.0, 5.0, 5.0, 1.25, 0.5],
    help="MPAIL raw action limits used for clipped-teacher/random modes.",
)
parser.add_argument(
    "--modes",
    nargs="+",
    default=["teacher_raw", "teacher_mpaillimit", "zero", "random_mpaillimit"],
    choices=["teacher_raw", "teacher_mpaillimit", "zero", "random_mpaillimit"],
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402


def _policy_obs(obs) -> torch.Tensor:
    return obs if torch.is_tensor(obs) else obs["policy"]


def _progress(env):
    return env.unwrapped.reward_manager.get_term_cfg("progress_context").func


def _obs_xyz_distance(obs_t: torch.Tensor) -> torch.Tensor:
    # Latest frame of insertive_asset_in_receptive_asset_frame is at indices 24:30.
    return obs_t[..., 24:27].norm(dim=-1)


def _term_mask(env, name: str, device: str) -> torch.Tensor | None:
    manager = getattr(env.unwrapped, "termination_manager", None)
    if manager is None or name not in getattr(manager, "active_terms", []):
        return None
    try:
        value = manager.get_term(name)
    except Exception:
        return None
    if torch.is_tensor(value):
        return value.to(device=device).bool()
    return torch.as_tensor(value, device=device).bool()


def _make_env(reset_type: str, seed: int, device: str, num_envs: int):
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=num_envs)
    env_cfg.seed = seed
    reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
    params = getattr(reset_event, "params", None)
    if params is None:
        raise ValueError("Expected reset_from_reset_states params in OmniReset env cfg.")
    params["reset_types"] = [reset_type]
    params["probs"] = [1.0]
    return gym.make(args_cli.task, cfg=env_cfg)


def _select_action(mode: str, policy, obs_t: torch.Tensor, raw_limit: torch.Tensor) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros((obs_t.shape[0], 7), device=obs_t.device)
    if mode == "random_mpaillimit":
        return (torch.rand((obs_t.shape[0], 7), device=obs_t.device) * 2.0 - 1.0) * raw_limit

    raw = policy(obs_t)
    if mode == "teacher_raw":
        return raw
    if mode == "teacher_mpaillimit":
        return (raw / raw_limit).clamp(-1.0, 1.0) * raw_limit
    raise ValueError(mode)


def _run_mode(reset_type: str, mode: str, seed: int, policy, raw_limit: torch.Tensor, device: str) -> dict:
    env = _make_env(reset_type, seed, device, args_cli.num_envs)
    obs, _ = env.reset()
    obs_t = _policy_obs(obs).to(device)

    initial_xyz = _obs_xyz_distance(obs_t).detach().clone().to(device)
    min_xyz = initial_xyz.clone()
    final_xyz = initial_xyz.clone()
    success_seen = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    done_seen = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    abnormal_seen = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    timeout_seen = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    steps_alive = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device)
    reward_sum = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device)
    action_norm_sum = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device)

    for _step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        active = ~done_seen
        if not bool(active.any()):
            break

        with torch.inference_mode():
            actions = _select_action(mode, policy, obs_t, raw_limit)
        next_obs, rewards, terms, truncs, _infos = env.step(actions)

        progress = _progress(env)
        next_obs_t = _policy_obs(next_obs).to(device)
        xyz = _obs_xyz_distance(next_obs_t).detach().to(device)
        success = progress.success.detach().to(device).bool()
        dones = (terms | truncs).to(device).bool()

        active_after_step = active
        min_xyz[active_after_step] = torch.minimum(min_xyz[active_after_step], xyz[active_after_step])
        final_xyz[active_after_step] = xyz[active_after_step]
        success_seen[active_after_step] |= success[active_after_step]
        reward_sum[active_after_step] += rewards.to(device)[active_after_step]
        action_norm_sum[active_after_step] += actions.norm(dim=-1)[active_after_step]
        steps_alive[active_after_step] += 1.0

        abnormal = _term_mask(env, "abnormal_robot", device)
        if abnormal is not None:
            abnormal_seen[active_after_step] |= abnormal[active_after_step]
        timeout = _term_mask(env, "time_out", device)
        if timeout is not None:
            timeout_seen[active_after_step] |= timeout[active_after_step]

        done_seen |= dones
        obs_t = next_obs_t

    env.close()

    active_steps = steps_alive.clamp_min(1.0)
    return {
        "reset_type": reset_type,
        "mode": mode,
        "seed": seed,
        "num_envs": args_cli.num_envs,
        "max_steps": args_cli.max_steps,
        "success_rate": success_seen.float().mean().item(),
        "done_rate": done_seen.float().mean().item(),
        "timeout_rate": timeout_seen.float().mean().item(),
        "abnormal_rate": abnormal_seen.float().mean().item(),
        "initial_xyz_mean": initial_xyz.mean().item(),
        "final_xyz_mean": final_xyz.mean().item(),
        "min_xyz_mean": min_xyz.mean().item(),
        "delta_final_xyz_mean": (final_xyz - initial_xyz).mean().item(),
        "delta_min_xyz_mean": (min_xyz - initial_xyz).mean().item(),
        "reward_sum_mean": reward_sum.mean().item(),
        "action_norm_mean": (action_norm_sum / active_steps).mean().item(),
        "steps_mean": steps_alive.mean().item(),
    }


def main() -> None:
    device = args_cli.device or "cuda"
    policy = torch.jit.load(str(args_cli.policy), map_location=device).eval()
    raw_limit = torch.as_tensor(args_cli.raw_action_limit, dtype=torch.float32, device=device).view(1, 7)

    rows = []
    for reset_type in args_cli.reset_types:
        for i, mode in enumerate(args_cli.modes):
            row = _run_mode(reset_type, mode, args_cli.seed + i, policy, raw_limit, device)
            rows.append(row)
            print(
                f"{reset_type:28s} {mode:18s} "
                f"success={row['success_rate']:.3f} "
                f"init={row['initial_xyz_mean']:.4f} final={row['final_xyz_mean']:.4f} "
                f"min={row['min_xyz_mean']:.4f} d_final={row['delta_final_xyz_mean']:.4f} "
                f"abnormal={row['abnormal_rate']:.3f} timeout={row['timeout_rate']:.3f} "
                f"act_norm={row['action_norm_mean']:.2f}",
                flush=True,
            )

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {args_cli.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
