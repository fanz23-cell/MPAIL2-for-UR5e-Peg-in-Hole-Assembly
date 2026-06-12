#!/usr/bin/env python3
"""Render a checkpoint sequence from an IsaacLab camera sensor.

This avoids both desktop screen recording and Gym's env.render() RGB annotator
path. The latter currently fails on this Isaac Sim install inside
omni.syntheticdata when attaching the rgb annotator.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--demo-path", type=Path, required=True)
parser.add_argument("--steps-per-checkpoint", type=int, default=120)
parser.add_argument("--frame-stride", type=int, default=1)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0")
parser.add_argument("--compact-obs-dim", type=int, choices=(13, 19, 22), default=22)
parser.add_argument("--raw-action-limit", nargs=7, type=float, default=None)
parser.add_argument("--force-gripper-close", action="store_true")
parser.add_argument("--gripper-close-value", type=float, default=-1.0)
parser.add_argument("--reset-types", nargs="+", default=None)
parser.add_argument("--camera-name", choices=("front_camera", "side_camera", "wrist_camera"), default="side_camera")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--keep-frames", action="store_true")
parser.add_argument("--print-every", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402

from mpail2.envs.isaac.omnireset_wrapper import (  # noqa: E402
    DEFAULT_RAW_ACTION_LIMIT,
    OmniResetMPAILWrapper,
    compact_omnireset_demonstrations,
)
from mpail2.learner import MPAIL2Learner  # noqa: E402
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig  # noqa: E402
from mpail2.train.utils.demos import load_demonstrations  # noqa: E402
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env  # noqa: E402


def _load_action_scale(ckpt: dict, device: str) -> torch.Tensor | None:
    if "action_scale" not in ckpt:
        return None
    return torch.as_tensor(ckpt["action_scale"], dtype=torch.float32, device=device).view(1, -1)


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 4:
        return frame[:, :, :3]
    return frame[:, :, :3]


def _official_camera_cfg(name: str) -> TiledCameraCfg:
    camera_cfgs = {
        "front_camera": TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
            update_period=0,
            height=args_cli.height,
            width=args_cli.width,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.0770121, -0.1679045, 0.4486344),
                rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
        ),
        "side_camera": TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
            update_period=0,
            height=args_cli.height,
            width=args_cli.width,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.8323904, 0.5877843, 0.2805111),
                rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=20.10),
        ),
        "wrist_camera": TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
            update_period=0,
            height=args_cli.height,
            width=args_cli.width,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0182505, -0.00408447, -0.0689107),
                rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
        ),
    }
    return camera_cfgs[name]


def _camera_frame(env, camera_name: str) -> np.ndarray | None:
    camera = env.unwrapped.scene[camera_name]
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        return None
    frame = rgb[0].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return _to_rgb(frame)


def main() -> None:
    device = args_cli.device or "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    if args_cli.reset_types:
        reset_event = getattr(getattr(env_cfg, "events", None), "reset_from_reset_states", None)
        params = getattr(reset_event, "params", None)
        if params is None:
            raise ValueError("Requested --reset-types, but env_cfg has no reset_from_reset_states params.")
        params["reset_types"] = list(args_cli.reset_types)
        params["probs"] = [1.0 / len(args_cli.reset_types)] * len(args_cli.reset_types)
        print(f"[INFO] Reset override: {params['reset_types']} probs={params['probs']}", flush=True)
    setattr(env_cfg.scene, args_cli.camera_name, _official_camera_cfg(args_cli.camera_name))
    print(f"[INFO] Added IsaacLab TiledCamera sensor: {args_cli.camera_name}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    scene_keys = list(env.unwrapped.scene.keys())
    print(f"[INFO] Scene keys: {scene_keys}", flush=True)
    if args_cli.camera_name not in scene_keys:
        raise RuntimeError(
            f"Camera {args_cli.camera_name!r} was not added to the scene. "
            f"Available scene keys: {scene_keys}"
        )
    camera = env.unwrapped.scene[args_cli.camera_name]
    print(
        "[INFO] Camera sensor ready: "
        f"{args_cli.camera_name} cfg_height={getattr(camera.cfg, 'height', '?')} "
        f"cfg_width={getattr(camera.cfg, 'width', '?')}",
        flush=True,
    )
    raw_action_limit = args_cli.raw_action_limit or list(DEFAULT_RAW_ACTION_LIMIT)
    env = OmniResetMPAILWrapper(
        env,
        raw_action_limit=raw_action_limit,
        compact_obs_dim=args_cli.compact_obs_dim,
        force_gripper_close=args_cli.force_gripper_close,
        gripper_close_value=args_cli.gripper_close_value,
    )
    print(
        "[INFO] Compact wrapper: "
        f"obs={args_cli.compact_obs_dim} raw_action_limit={raw_action_limit} "
        f"force_gripper_close={args_cli.force_gripper_close}",
        flush=True,
    )

    learner_cfg = sync_learner_dims_from_env(OmniResetStateLearnerConfig(), env)
    demonstrations = load_demonstrations(str(args_cli.demo_path), device=device)
    demonstrations = compact_omnireset_demonstrations(
        demonstrations,
        include_ee_to_hole=args_cli.compact_obs_dim in (19, 22),
        include_ee_vel=args_cli.compact_obs_dim == 22,
    )
    learner = MPAIL2Learner(demonstrations, args_cli.num_envs, learner_cfg, device=device)
    learner.init_storage(
        num_envs=args_cli.num_envs,
        num_steps_per_env=env.unwrapped.max_episode_length,
        actor_obs_shape={"policy": (learner_cfg.planner_cfg.encoder_cfg.input_dim,)},
        critic_obs_shape=None,
        action_shape=[learner_cfg.planner_cfg.action_dim],
    )
    learner.eval()

    tmp_dir = Path(tempfile.mkdtemp(prefix="mpail_internal_viewport_frames_"))
    frames: list[np.ndarray] = []
    frame_idx = 0
    try:
        for ckpt_iter in args_cli.checkpoints:
            ckpt_path = args_cli.checkpoint_dir / f"model_{ckpt_iter}.pt"
            if not ckpt_path.exists():
                print(f"[SKIP] missing {ckpt_path}", flush=True)
                continue
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            learner.planner.load_state_dict(ckpt["model_state_dict"])
            action_scale = _load_action_scale(ckpt, device)
            print(f"[CHECKPOINT_START] model_{ckpt_iter}.pt iter={ckpt.get('iter', '?')}", flush=True)

            obs, _ = env.reset()
            for step in range(args_cli.steps_per_checkpoint):
                with torch.inference_mode():
                    actions = learner.act(obs)
                env_actions = actions if action_scale is None else actions * action_scale
                obs, rewards, terms, truncs, _ = env.step(env_actions.to(env.unwrapped.device))
                if step % args_cli.frame_stride == 0:
                    frame = _camera_frame(env, args_cli.camera_name)
                    if frame is not None:
                        png_path = tmp_dir / f"frame_{frame_idx:06d}.png"
                        imageio.imwrite(png_path, frame)
                        frames.append(frame)
                        frame_idx += 1
                dones = terms | truncs
                if step % args_cli.print_every == 0 or dones.any():
                    print(
                        f"[CHECKPOINT_STEP] iter={ckpt_iter} step={step} "
                        f"reward={rewards.mean().item():.4f} done={bool(dones.any())}",
                        flush=True,
                    )
                if dones.any():
                    obs, _ = env.reset()
            print(f"[CHECKPOINT_END] model_{ckpt_iter}.pt", flush=True)
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No viewport frames were captured.")
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(args_cli.output), frames, fps=args_cli.fps, macro_block_size=1)
    print(f"[INFO] Internal viewport video saved: {args_cli.output} ({len(frames)} frames)", flush=True)
    if args_cli.keep_frames:
        print(f"[INFO] Kept frames under: {tmp_dir}", flush=True)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
