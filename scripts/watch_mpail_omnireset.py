#!/usr/bin/env python3
"""Watch a trained MPAIL2 OmniReset checkpoint live in Isaac Lab.

This script does not record video. Run without ``--headless`` to open the
Isaac Sim viewer and watch the policy step the environment.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Watch a trained MPAIL2 checkpoint on OmniReset.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--demo-path",
    type=Path,
    default=Path("mpail2/train/isaac_franka/demos/peg_expert_rl_mpail_filtered.pt"),
    help="MPAIL demo file used to reconstruct the fixed observation normalizer.",
)
parser.add_argument("--num-steps", type=int, default=300)
parser.add_argument("--print-every", type=int, default=10)
parser.add_argument("--sleep", type=float, default=0.0, help="Optional wall-clock pause between env steps.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument(
    "--policy-only",
    action="store_true",
    help="Bypass MPAIL planning and execute the learned encoder+policy mean action directly.",
)
parser.add_argument(
    "--compare-policy",
    action="store_true",
    help="Log the policy-only action beside the MPPI-selected action at the same state.",
)
parser.add_argument(
    "--csv-path",
    type=Path,
    default=None,
    help="Optional CSV path for headless numeric rollout diagnostics.",
)
parser.add_argument(
    "--reset-types",
    nargs="+",
    default=None,
    help="Optional OmniReset reset_types override, e.g. ObjectPartiallyAssembledEEGrasped",
)
parser.add_argument(
    "--compact-wrapper",
    action="store_true",
    help="Use the compact OmniReset MPAIL wrapper used by compact-wrapper training.",
)
parser.add_argument(
    "--compact-obs-dim",
    type=int,
    choices=(13, 19, 22, 215),
    default=19,
    help="Observation dimension. Use 13 legacy, 19 ee_to_hole, 22 ee_to_hole+ee_vel, or 215 raw policy obs.",
)
parser.add_argument(
    "--raw-action-limit",
    nargs=7,
    type=float,
    default=None,
    help="Raw OSC action limits for --compact-wrapper.",
)
parser.add_argument(
    "--force-gripper-close",
    action="store_true",
    help="Override the compact-wrapper gripper action to a constant close command.",
)
parser.add_argument(
    "--gripper-close-value",
    type=float,
    default=-1.0,
    help="Normalized gripper close value used with --force-gripper-close.",
)
parser.add_argument(
    "--held-distance-threshold",
    type=float,
    default=0.08,
    help="Heuristic ee-to-peg distance threshold for printing whether the object is still held.",
)
parser.add_argument(
    "--require-held-reset",
    action="store_true",
    help="Keep resetting until the initial ee-to-peg distance passes --held-distance-threshold.",
)
parser.add_argument(
    "--max-held-reset-attempts",
    type=int,
    default=50,
    help="Maximum reset attempts when --require-held-reset is enabled.",
)
parser.add_argument(
    "--held-reset-settle-steps",
    type=int,
    default=5,
    help="Zero-action settle steps before checking a held reset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Keep downstream OmniKit parsers from seeing script args twice.
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402

from mpail2.learner import MPAIL2Learner  # noqa: E402
from mpail2.envs.isaac.omnireset_wrapper import (  # noqa: E402
    DEFAULT_RAW_ACTION_LIMIT,
    OmniResetMPAILWrapper,
    compact_omnireset_demonstrations,
)
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig  # noqa: E402
from mpail2.train.utils.demos import load_demonstrations  # noqa: E402
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env  # noqa: E402


def _policy_first_action_mean(policy, z: torch.Tensor) -> torch.Tensor:
    out = policy.model(z)
    mu, _log_std = torch.chunk(out, 2, dim=-1)
    mu = mu.view(z.shape[0], policy.horizon, policy.action_dim)
    return torch.tanh(mu[:, 0])


def _act_policy_only(learner: MPAIL2Learner, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    if learner._obs_normalizer is not None:
        obs = learner._obs_normalizer(obs)
    z = learner._encoder(obs)
    return _policy_first_action_mean(learner._policy, z)


def _load_action_scale(ckpt: dict, device: str) -> torch.Tensor | None:
    if "action_scale" not in ckpt:
        return None
    return torch.as_tensor(ckpt["action_scale"], dtype=torch.float32, device=device).view(1, -1)


def _held_debug_from_obs(obs: dict[str, torch.Tensor], threshold: float) -> tuple[float | None, bool | None]:
    policy_obs = obs.get("policy") if isinstance(obs, dict) else None
    if policy_obs is None or policy_obs.shape[-1] < 9:
        return None, None
    ee_to_peg = policy_obs[..., 6:9]
    ee_peg_dist = ee_to_peg.norm(dim=-1).mean().item()
    return ee_peg_dist, ee_peg_dist < threshold


def _reset_until_held(env, threshold: float, max_attempts: int, settle_steps: int):
    """Reset until the compact observation indicates the peg is near the EE."""

    last_obs = None
    for attempt in range(1, max_attempts + 1):
        obs, info = env.reset()
        zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(settle_steps):
            obs, _, terminated, truncated, info = env.step(zero_action)
            if bool((terminated | truncated).any()):
                obs, info = env.reset()
        last_obs = obs
        ee_peg_dist, held = _held_debug_from_obs(obs, threshold)
        if held:
            print(
                f"[INFO] Held reset accepted after {attempt} attempt(s): ee_peg={ee_peg_dist:.4f}",
                flush=True,
            )
            return obs, info
        print(
            f"[INFO] Held reset rejected attempt {attempt}/{max_attempts}: ee_peg={ee_peg_dist}",
            flush=True,
        )
    raise RuntimeError(
        f"Could not find a held reset in {max_attempts} attempts. "
        f"Last obs present={last_obs is not None}; try increasing --held-distance-threshold."
    )


def main() -> None:
    device = args_cli.device or "cuda"

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42  # Fixed seed for consistent reset positions
    if args_cli.reset_types:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested --reset-types, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = list(args_cli.reset_types)
        params["probs"] = [1.0 / len(args_cli.reset_types)] * len(args_cli.reset_types)
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if args_cli.compact_wrapper:
        raw_action_limit = args_cli.raw_action_limit or list(DEFAULT_RAW_ACTION_LIMIT)
        env = OmniResetMPAILWrapper(
            env,
            raw_action_limit=raw_action_limit,
            compact_obs_dim=args_cli.compact_obs_dim,
            force_gripper_close=args_cli.force_gripper_close,
            gripper_close_value=args_cli.gripper_close_value,
        )
        print(
            "[INFO] Enabled compact OmniReset wrapper: "
            f"compact_obs={args_cli.compact_obs_dim} raw_action_limit={raw_action_limit} "
            f"force_gripper_close={args_cli.force_gripper_close}"
        )

    learner_cfg = sync_learner_dims_from_env(OmniResetStateLearnerConfig(), env)
    demonstrations = load_demonstrations(str(args_cli.demo_path), device=device)
    if args_cli.compact_wrapper and args_cli.compact_obs_dim != 215:
        demonstrations = compact_omnireset_demonstrations(
            demonstrations,
            include_ee_to_hole=args_cli.compact_obs_dim in (19, 22),
            include_ee_vel=args_cli.compact_obs_dim == 22,
        )
        print(f"[INFO] Compact demos: policy shape={tuple(demonstrations['policy'].shape)}")
    learner = MPAIL2Learner(demonstrations, args_cli.num_envs, learner_cfg, device=device)
    if not args_cli.policy_only:
        learner.init_storage(
            num_envs=args_cli.num_envs,
            num_steps_per_env=env.unwrapped.max_episode_length,
            actor_obs_shape={"policy": (learner_cfg.planner_cfg.encoder_cfg.input_dim,)},
            critic_obs_shape=None,
            action_shape=[learner_cfg.planner_cfg.action_dim],
        )

    ckpt = torch.load(str(args_cli.checkpoint), map_location="cpu", weights_only=False)
    learner.planner.load_state_dict(ckpt["model_state_dict"])
    action_scale = _load_action_scale(ckpt, device)
    learner.eval()
    print(f"[INFO] Loaded checkpoint iter={ckpt.get('iter', '?')}")
    if action_scale is not None:
        print(f"[INFO] Scaling actions before env.step: {action_scale.detach().cpu().tolist()}")
    mode = "policy-only encoder+policy mean" if args_cli.policy_only else "MPAIL planner"
    print(f"[INFO] Stepping {mode}. Press Ctrl+C to stop.")

    if args_cli.require_held_reset:
        obs, _ = _reset_until_held(
            env,
            args_cli.held_distance_threshold,
            args_cli.max_held_reset_attempts,
            args_cli.held_reset_settle_steps,
        )
    else:
        obs, _ = env.reset()
    total_reward = torch.zeros(args_cli.num_envs, device=device)
    csv_file = None
    csv_writer = None
    if args_cli.csv_path is not None:
        args_cli.csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args_cli.csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step",
                "reward_mean",
                "total_reward_mean",
                "terminated_mean",
                "truncated_mean",
                "action_norm",
                "env_action_norm",
                "policy_action_norm",
                "mppi_policy_delta_norm",
                "success_rate",
                "xyz_distance",
                "ee_peg_dist",
                "held",
            ],
        )
        csv_writer.writeheader()

    try:
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break

            with torch.inference_mode():
                policy_actions = None
                if args_cli.compare_policy or args_cli.policy_only:
                    policy_actions = _act_policy_only(learner, obs)
                if args_cli.policy_only:
                    actions = policy_actions
                else:
                    actions = learner.act(obs)

            env_actions = actions if action_scale is None else actions * action_scale
            obs, rewards, terms, truncs, _infos = env.step(env_actions.to(env.unwrapped.device))
            rewards = rewards.to(device)
            dones = terms | truncs
            total_reward += rewards
            progress = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
            success_rate = progress.success.float().mean().item()
            xyz_distance = progress.xyz_distance.mean().item()
            action_norm = actions.norm(dim=-1).mean().item()
            env_action_norm = env_actions.norm(dim=-1).mean().item()
            policy_action_norm = None
            mppi_policy_delta_norm = None
            if policy_actions is not None:
                policy_action_norm = policy_actions.norm(dim=-1).mean().item()
                mppi_policy_delta_norm = (actions - policy_actions).norm(dim=-1).mean().item()
            ee_peg_dist, held = _held_debug_from_obs(obs, args_cli.held_distance_threshold)

            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "step": step,
                        "reward_mean": rewards.mean().item(),
                        "total_reward_mean": total_reward.mean().item(),
                        "terminated_mean": terms.float().mean().item(),
                        "truncated_mean": truncs.float().mean().item(),
                        "action_norm": action_norm,
                        "env_action_norm": env_action_norm,
                        "policy_action_norm": policy_action_norm,
                        "mppi_policy_delta_norm": mppi_policy_delta_norm,
                        "success_rate": success_rate,
                        "xyz_distance": xyz_distance,
                        "ee_peg_dist": ee_peg_dist,
                        "held": held,
                    }
                )

            if step % args_cli.print_every == 0 or dones.any():
                held_msg = ""
                if ee_peg_dist is not None:
                    held_msg = f" ee_peg={ee_peg_dist:.4f} held~={held}"
                compare_msg = ""
                if policy_action_norm is not None and mppi_policy_delta_norm is not None:
                    compare_msg = (
                        f" policy_action_norm={policy_action_norm:.4f} "
                        f"mppi_policy_delta={mppi_policy_delta_norm:.4f}"
                    )
                print(
                    f"step={step:04d} reward={rewards.mean().item():.4f} "
                    f"total_reward={total_reward.mean().item():.4f} "
                    f"action_norm={action_norm:.4f} env_action_norm={env_action_norm:.4f} "
                    f"{compare_msg} "
                    f"success={success_rate:.3f} xyz={xyz_distance:.4f} "
                    f"done={bool(dones.any())}{held_msg}",
                    flush=True,
                )

            if dones.any():
                # Reset env seed to 42 before each reset to ensure consistent positions
                env.unwrapped.cfg.seed = 42
                if args_cli.require_held_reset:
                    obs, _ = _reset_until_held(
                        env,
                        args_cli.held_distance_threshold,
                        args_cli.max_held_reset_attempts,
                        args_cli.held_reset_settle_steps,
                    )
                else:
                    obs, _ = env.reset()
                total_reward.zero_()

            if args_cli.sleep > 0:
                time.sleep(args_cli.sleep)
    finally:
        if csv_file is not None:
            csv_file.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
