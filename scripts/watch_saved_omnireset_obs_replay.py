#!/usr/bin/env python3
"""Visualize saved OmniReset rollout observations by writing poses into Isaac Sim."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Watch saved OmniReset expert observations.")
parser.add_argument("--rollouts", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to replay; 0 means all.")
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--frame-stride", type=int, default=1)
parser.add_argument("--print-every", type=int, default=40)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--reset-type", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402


def _axis_angle_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1)
    axis = axis_angle / angle.clamp_min(1e-9).unsqueeze(-1)
    quat = math_utils.quat_from_angle_axis(angle, axis)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where((angle > 1e-8).unsqueeze(-1), quat, identity)


def _pose_from_obs(obs: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
    pos = obs[..., start : start + 3]
    quat = _axis_angle_to_quat(obs[..., start + 3 : start + 6])
    return pos, quat


def _write_saved_frame(env, obs_row: torch.Tensor) -> None:
    device = env.unwrapped.device
    scene = env.unwrapped.scene
    robot = scene["robot"]
    insertive = scene["insertive_object"]
    receptive = scene["receptive_object"]

    obs_row = obs_row.to(device=device, dtype=torch.float32).view(1, -1)
    joint_pos = obs_row[:, 113:125]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)

    env.unwrapped.sim.forward()
    scene.update(float(env.unwrapped.step_dt))

    wrist_idx = robot.body_names.index("wrist_3_link")
    wrist_pos = robot.data.body_link_pos_w[:, wrist_idx]
    wrist_quat = robot.data.body_link_quat_w[:, wrist_idx]

    ins_pos_b, ins_quat_b = _pose_from_obs(obs_row, 179)
    rec_pos_b, rec_quat_b = _pose_from_obs(obs_row, 209)
    ins_pos_w, ins_quat_w = math_utils.combine_frame_transforms(wrist_pos, wrist_quat, ins_pos_b, ins_quat_b)
    rec_pos_w, rec_quat_w = math_utils.combine_frame_transforms(wrist_pos, wrist_quat, rec_pos_b, rec_quat_b)

    zero_vel = torch.zeros((1, 6), device=device)
    insertive.write_root_pose_to_sim(torch.cat([ins_pos_w, ins_quat_w], dim=-1))
    insertive.write_root_velocity_to_sim(zero_vel)
    receptive.write_root_pose_to_sim(torch.cat([rec_pos_w, rec_quat_w], dim=-1))
    receptive.write_root_velocity_to_sim(zero_vel)

    env.unwrapped.sim.forward()
    scene.update(float(env.unwrapped.step_dt))


def main() -> None:
    device = args_cli.device or "cuda"
    bundle = torch.load(str(args_cli.rollouts), map_location="cpu", weights_only=False)
    data = bundle["data"]
    metadata = bundle.get("metadata", {})
    obs = data["obs"]["policy"].float()
    episode_ids = data["episode_id"].long()
    success = data.get("success")
    xyz = data.get("xyz_distance")

    unique_ids = torch.unique_consecutive(episode_ids).tolist()
    if args_cli.episodes and args_cli.episodes > 0:
        unique_ids = unique_ids[: args_cli.episodes]

    reset_type = args_cli.reset_type
    if reset_type is None:
        reset_type = metadata.get("reset_type")
    print(
        f"[INFO] Observation-replaying {len(unique_ids)} saved episodes from {args_cli.rollouts} "
        f"reset_type={reset_type!r}",
        flush=True,
    )
    print(f"[INFO] Metadata: {metadata}", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env_cfg.seed = args_cli.seed
    if reset_type:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is not None:
            params["reset_types"] = [reset_type]
            params["probs"] = [1.0]
            print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        env.reset()
        for ep_idx, ep_id in enumerate(unique_ids, start=1):
            if not simulation_app.is_running():
                break
            indices = (episode_ids == int(ep_id)).nonzero(as_tuple=False).flatten()
            print(
                f"[EPISODE_START] {ep_idx}/{len(unique_ids)} saved_episode_id={int(ep_id)} "
                f"frames={len(indices)} success_any={bool(success[indices].any()) if success is not None else '?'}",
                flush=True,
            )
            for local_step, idx in enumerate(indices.tolist()):
                if not simulation_app.is_running():
                    break
                if local_step % max(1, args_cli.frame_stride) != 0:
                    continue
                _write_saved_frame(env, obs[idx])
                env.unwrapped.sim.render()
                if local_step % args_cli.print_every == 0 or local_step == len(indices) - 1:
                    print(
                        f"[EP {ep_idx:02d}/{len(unique_ids):02d}] frame={local_step:03d}/{len(indices)} "
                        f"saved_success={bool(success[idx]) if success is not None else None} "
                        f"saved_xyz={float(xyz[idx]) if xyz is not None else None}",
                        flush=True,
                    )
                if args_cli.sleep > 0:
                    time.sleep(args_cli.sleep)
            print(f"[EPISODE_END] {ep_idx}/{len(unique_ids)} saved_episode_id={int(ep_id)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
