#!/usr/bin/env python3
"""Check OmniReset Robotiq gripper action sign in Isaac Lab."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug OmniReset gripper open/close sign.")
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--reset-type", default="ObjectAnywhereEEGrasped")
parser.add_argument("--steps", type=int, default=8)
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


def _print_state(env, label: str, action_term) -> None:
    robot = env.unwrapped.scene["robot"]
    finger_ids, finger_names = robot.find_joints(["finger_joint"])
    finger_pos = robot.data.joint_pos[:, finger_ids].detach().cpu().flatten().tolist()
    raw = getattr(action_term, "raw_actions", getattr(action_term, "_raw_actions", None))
    processed = getattr(action_term, "processed_actions", getattr(action_term, "_processed_actions", None))
    raw_list = raw.detach().cpu().flatten().tolist() if raw is not None else None
    processed_list = processed.detach().cpu().flatten().tolist() if processed is not None else None
    print(
        f"{label}: finger_names={finger_names} finger_joint={finger_pos} "
        f"raw={raw_list} processed={processed_list}",
        flush=True,
    )


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    _set_reset_type(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        env.reset()
        gripper_term = env.unwrapped.action_manager._terms["gripper"]
        _print_state(env, "after reset", gripper_term)

        for sign in (1.0, -1.0, 1.0, -1.0):
            action = torch.zeros((1, 7), device=env.unwrapped.device)
            action[:, -1] = sign
            for step in range(args_cli.steps):
                env.step(action)
                _print_state(env, f"gripper_action={sign:+.1f} step={step + 1}", gripper_term)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
