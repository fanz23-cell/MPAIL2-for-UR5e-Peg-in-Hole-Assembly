#!/usr/bin/env python3
"""Check OmniReset OSC arm action signs in Isaac Lab."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug OmniReset arm action signs.")
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--reset-type", default="ObjectAnywhereEEGrasped")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--magnitude", type=float, default=1.0)
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


def _set_reset_type(env_cfg) -> None:
    reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
    params = getattr(reset_event, "params", None)
    if params is None:
        raise ValueError("Task has no reset_from_reset_states params.")
    params["reset_types"] = [args_cli.reset_type]
    params["probs"] = [1.0]


def _ee_pos_b(env) -> torch.Tensor:
    arm_term = env.unwrapped.action_manager._terms["arm"]
    pos, _quat = arm_term._get_ee_pose_root_frame()
    return pos.detach().clone()


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    _set_reset_type(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        names = ["x", "y", "z", "roll", "pitch", "yaw"]
        for dim, name in enumerate(names):
            env.reset()
            start = _ee_pos_b(env)
            action = torch.zeros((1, 7), device=env.unwrapped.device)
            action[:, dim] = args_cli.magnitude
            action[:, -1] = -1.0  # keep gripper closed
            for _ in range(args_cli.steps):
                env.step(action)
            end = _ee_pos_b(env)
            arm_term = env.unwrapped.action_manager._terms["arm"]
            raw = arm_term.raw_actions.detach().cpu().flatten().tolist()
            processed = arm_term.processed_actions.detach().cpu().flatten().tolist()
            print(
                f"dim={dim} {name} raw={raw} processed={processed} "
                f"ee_start={start.cpu().flatten().tolist()} "
                f"ee_end={end.cpu().flatten().tolist()} "
                f"delta={(end - start).cpu().flatten().tolist()}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
