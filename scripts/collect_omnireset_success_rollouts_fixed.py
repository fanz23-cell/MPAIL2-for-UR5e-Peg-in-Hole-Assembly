#!/usr/bin/env python3
"""Collect successful OmniReset rollouts from an exported Stage-2 teacher.

This is the generalized version of the route-A collector.  It can collect from
one explicit reset type, such as ``ObjectAnywhereEEGrasped`` or
``ObjectAnywhereEEAnywhere``, and stores complete successful episodes with
per-transition success/reward/done diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect successful OmniReset state/action rollouts.")
parser.add_argument("--policy", type=Path, required=True, help="TorchScript policy.pt exported from Stage-2 teacher")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--target-success-episodes", type=int, default=52)
parser.add_argument("--max-env-steps", type=int, default=60000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--clip-actions", type=float, default=0.0, help="Clip raw actions if >0; 0 matches exported policy.")
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument(
    "--reset-type",
    default=None,
    help="Optional single OmniReset reset type override. If omitted, env default reset mixture is used.",
)
parser.add_argument(
    "--fixed-position",
    action="store_true",
    help="Force all resets to use exactly the same fixed position (overrides random ranges).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402

# Import fixed position override
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mpail2.envs.isaac.fixed_position_reset import apply_fixed_position_reset_override  # noqa: E402


def _policy_obs(obs) -> torch.Tensor:
    return obs if isinstance(obs, torch.Tensor) else obs["policy"]


def _progress_term(env):
    return env.unwrapped.reward_manager.get_term_cfg("progress_context").func


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    if args_cli.reset_type:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested --reset-type, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = [args_cli.reset_type]
        params["probs"] = [1.0]
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}")
    else:
        print("[INFO] Using task default reset distribution.")

    # Apply fixed position override if requested
    if args_cli.fixed_position:
        print("[INFO] *** APPLYING FIXED POSITION OVERRIDE ***")
        events = getattr(env_cfg, "events", None)
        if events:
            print(f"[DEBUG] Found events object: {type(events)}")
            # List all event attributes
            for attr in dir(events):
                if not attr.startswith('_') and 'reset' in attr.lower():
                    print(f"[DEBUG]   Event attribute: {attr}")

        apply_fixed_position_reset_override(env_cfg)
        print("[INFO] *** FIXED POSITION MODE *** All 55 episodes will have IDENTICAL start positions!")
    else:
        print("[INFO] Fixed position mode NOT enabled (--fixed-position flag not set)")

    env = gym.make(args_cli.task, cfg=env_cfg)
    policy = torch.jit.load(str(args_cli.policy), map_location=device).eval()

    obs, _ = env.reset()

    # If fixed position mode, manually set object positions after reset
    if args_cli.fixed_position:
        import torch
        # Fixed positions (in world frame, relative to metal support)
        fixed_peg_pos = torch.tensor([[0.45, 0.10, 0.15]], device=device)  # x, y, z
        fixed_peg_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # w, x, y, z

        fixed_hole_pos = torch.tensor([[0.50, 0.00, 0.10]], device=device)  # x, y, z
        fixed_hole_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # w, x, y, z

        # Set insertive object (peg) position
        insertive_obj = env.unwrapped.scene["insertive_object"]
        insertive_obj.write_root_pose_to_sim(
            torch.cat([fixed_peg_pos.repeat(args_cli.num_envs, 1), fixed_peg_quat.repeat(args_cli.num_envs, 1)], dim=1)
        )

        # Set receptive object (hole) position
        receptive_obj = env.unwrapped.scene["receptive_object"]
        receptive_obj.write_root_pose_to_sim(
            torch.cat([fixed_hole_pos.repeat(args_cli.num_envs, 1), fixed_hole_quat.repeat(args_cli.num_envs, 1)], dim=1)
        )

        # CRITICAL: Step the simulation a few times to let physics settle
        for _ in range(10):
            env.unwrapped.sim.step(render=False)

        # Re-read observations after physics settled
        obs = env.unwrapped._get_observations()

        print(f"[INFO] Forced peg position to: {fixed_peg_pos[0].tolist()}")
        print(f"[INFO] Forced hole position to: {fixed_hole_pos[0].tolist()}")

    obs_t = _policy_obs(obs)
    episode_buffers: list[list[dict[str, torch.Tensor | int | float | bool]]] = [[] for _ in range(args_cli.num_envs)]
    episode_ids = torch.arange(args_cli.num_envs, dtype=torch.long)
    next_episode_id = args_cli.num_envs

    kept: list[dict[str, torch.Tensor | int | float | bool]] = []
    total_episodes = 0
    success_episodes = 0
    env_steps = 0

    try:
        while (
            simulation_app.is_running()
            and success_episodes < args_cli.target_success_episodes
            and env_steps < args_cli.max_env_steps
        ):
            with torch.inference_mode():
                raw_actions = policy(obs_t.to(device))
                actions = raw_actions.clamp(-args_cli.clip_actions, args_cli.clip_actions) if args_cli.clip_actions > 0 else raw_actions

            next_obs, rewards, terms, truncs, _infos = env.step(actions)
            next_obs_t = _policy_obs(next_obs)
            dones = terms | truncs

            progress = _progress_term(env)
            success = progress.success.detach().clone()
            xyz_distance = progress.xyz_distance.detach().clone()
            action_norm = actions.norm(dim=-1).detach().clone()

            for env_id in range(args_cli.num_envs):
                episode_buffers[env_id].append(
                    {
                        "obs": obs_t[env_id].detach().cpu(),
                        "action": actions[env_id].detach().cpu(),
                        "next_obs": next_obs_t[env_id].detach().cpu(),
                        "reward": float(rewards[env_id].detach().cpu()),
                        "terminated": bool(terms[env_id].detach().cpu()),
                        "truncated": bool(truncs[env_id].detach().cpu()),
                        "done": bool(dones[env_id].detach().cpu()),
                        "success": bool(success[env_id].detach().cpu()),
                        "xyz_distance": float(xyz_distance[env_id].detach().cpu()),
                        "action_norm": float(action_norm[env_id].detach().cpu()),
                        "episode_id": int(episode_ids[env_id]),
                        "step_in_episode": len(episode_buffers[env_id]),
                    }
                )

            done_ids = dones.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            for env_id in done_ids:
                total_episodes += 1
                episode = episode_buffers[env_id]
                if any(bool(row["success"]) for row in episode):
                    kept.extend(episode)
                    success_episodes += 1
                episode_buffers[env_id] = []
                episode_ids[env_id] = next_episode_id
                next_episode_id += 1

            # If fixed position mode, force object positions after each reset
            if args_cli.fixed_position and done_ids:
                fixed_peg_pos = torch.tensor([[0.45, 0.10, 0.15]], device=device)
                fixed_peg_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
                fixed_hole_pos = torch.tensor([[0.50, 0.00, 0.10]], device=device)
                fixed_hole_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

                insertive_obj = env.unwrapped.scene["insertive_object"]
                insertive_obj.write_root_pose_to_sim(
                    torch.cat([fixed_peg_pos.repeat(args_cli.num_envs, 1), fixed_peg_quat.repeat(args_cli.num_envs, 1)], dim=1)
                )

                receptive_obj = env.unwrapped.scene["receptive_object"]
                receptive_obj.write_root_pose_to_sim(
                    torch.cat([fixed_hole_pos.repeat(args_cli.num_envs, 1), fixed_hole_quat.repeat(args_cli.num_envs, 1)], dim=1)
                )

                # CRITICAL: Step simulation to let physics settle
                for _ in range(10):
                    env.unwrapped.sim.step(render=False)

                # Re-read observations
                obs_t = _policy_obs(env.unwrapped._get_observations())

            env_steps += args_cli.num_envs
            obs_t = next_obs_t

            if env_steps % (args_cli.num_envs * 50) == 0 or done_ids:
                print(
                    f"env_steps={env_steps} total_episodes={total_episodes} "
                    f"success_episodes={success_episodes} kept_transitions={len(kept)}",
                    flush=True,
                )
    finally:
        env.close()

    if not kept:
        raise RuntimeError("No successful episodes collected; refusing to save an empty dataset.")

    def stack_tensor(key: str) -> torch.Tensor:
        return torch.stack([row[key] for row in kept])  # type: ignore[arg-type]

    output = {
        "data": {
            "obs": {"policy": stack_tensor("obs").float()},
            "actions": stack_tensor("action").float(),
            "next_obs": {"policy": stack_tensor("next_obs").float()},
            "rewards": torch.tensor([row["reward"] for row in kept], dtype=torch.float32),
            "terminated": torch.tensor([row["terminated"] for row in kept], dtype=torch.bool),
            "truncated": torch.tensor([row["truncated"] for row in kept], dtype=torch.bool),
            "dones": torch.tensor([row["done"] for row in kept], dtype=torch.bool),
            "success": torch.tensor([row["success"] for row in kept], dtype=torch.bool),
            "xyz_distance": torch.tensor([row["xyz_distance"] for row in kept], dtype=torch.float32),
            "action_norm": torch.tensor([row["action_norm"] for row in kept], dtype=torch.float32),
            "episode_id": torch.tensor([row["episode_id"] for row in kept], dtype=torch.long),
            "step_in_episode": torch.tensor([row["step_in_episode"] for row in kept], dtype=torch.long),
        },
        "metadata": {
            "policy": str(args_cli.policy),
            "task": args_cli.task,
            "reset_type": args_cli.reset_type,
            "target_success_episodes": args_cli.target_success_episodes,
            "total_episodes_seen": total_episodes,
            "success_episodes_kept": success_episodes,
            "note": "Complete successful episodes; downstream builder removes post-success idle.",
        },
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, str(args_cli.output))
    print(f"Saved {len(kept)} transitions from {success_episodes} successful episodes -> {args_cli.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
