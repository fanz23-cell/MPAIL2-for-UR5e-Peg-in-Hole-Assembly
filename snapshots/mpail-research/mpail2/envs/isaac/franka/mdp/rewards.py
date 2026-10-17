from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Per-env persistent stage trackers for pick-and-place sparse success.
# Keyed by id(env); cleared when an episode reset is detected.
_stage2_reached: dict = {}  # cube z > 0.15
_stage3_reached: dict = {}  # cube y < -0.1 (after stage 2)
_stage4_reached: dict = {}  # cube z < 0.03 and ee z < 0.03 (after stage 3)
_episode_length_tracker: dict = {}


def _ee_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_frame.data.target_pos_w[:, 0, :]
    )
    return pos_b


def _ee_to_object_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return torch.norm(object.data.root_pos_w[:, :3] - ee_frame.data.target_pos_w[:, 0, :], dim=1)


def _object_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object.data.root_pos_w[:, :3]
    )
    return pos_b


def push_reward(
    env: ManagerBasedRLEnv,
    y_threshold: float = -0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense: -(EE-to-block dist) - (block-to-y_threshold dist)."""
    ee_to_block = _ee_to_object_distance(env, object_cfg, ee_frame_cfg)
    obj_b = _object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    return -ee_to_block - torch.abs(obj_b[:, 1] - y_threshold)


def push_sparse_reward(
    env: ManagerBasedRLEnv,
    y_threshold: float = -0.1,
    reward_amount: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse +reward_amount when cube crosses y_threshold in robot frame."""
    obj_b = _object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    return torch.where(
        obj_b[:, 1] < y_threshold,
        torch.full_like(obj_b[:, 1], reward_amount),
        torch.zeros_like(obj_b[:, 1]),
    )


def pick_place_dense_reward(
    env: ManagerBasedRLEnv,
    y_target: float = -0.1,
    min_height: float = 0.02,
    height_gain: float = 0.75,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward: approach cube, encourage lift, then reduce lateral error to goal strip."""
    ee_to_block = _ee_to_object_distance(env, object_cfg, ee_frame_cfg)
    obj_b = _object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    lift = height_gain * torch.clamp(obj_b[:, 2] - min_height, min=0.0)
    return -ee_to_block - torch.abs(obj_b[:, 1] - y_target) + lift


_debug_call_counter: dict = {}
_debug_max_block_z: dict = {}  # per-env max of block_z since last reset


def pick_place_sparse_reward(
    env: ManagerBasedRLEnv,
    lift_height: float = 0.15,
    y_target: float = -0.1,
    place_height: float = 0.03,
    reward_amount: float = 1.0,
    require_ee_low: bool = True,
    debug: bool = False,
    debug_every: int = 200,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Sparse success for pick-and-place with four sequential stages.

    Stages (must be completed in order; state persists across steps until reset):
        Stage 1: cube z <= ``lift_height`` (not lifted)
        Stage 2: cube z > ``lift_height`` (lifted)
        Stage 3: stage 2 reached and cube y < ``y_target``
        Stage 4: stage 3 reached and cube z < ``place_height``
                 (and ee z < ``place_height`` if ``require_ee_low``)

    Returns ``reward_amount`` when stage 4 is reached, otherwise 0.

    Set ``debug=True`` to print per-stage counts and position extrema every
    ``debug_every`` calls (useful for tuning thresholds).
    """
    env_id = id(env)
    num_envs = env.num_envs
    device = env.device

    obj_b = _object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    ee_b = _ee_pos_in_robot_frame(env, robot_cfg, ee_frame_cfg)
    block_y, block_z = obj_b[:, 1], obj_b[:, 2]
    ee_z = ee_b[:, 2]

    is_reset = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if hasattr(env, "episode_length_buf"):
        cur_len = env.episode_length_buf
        prev_len = _episode_length_tracker.get(env_id)
        if prev_len is None:
            is_reset = cur_len == 0
        else:
            is_reset = (cur_len < prev_len) | ((cur_len == 0) & (prev_len > 0))
        _episode_length_tracker[env_id] = cur_len.clone()

    for stage_dict in (_stage2_reached, _stage3_reached, _stage4_reached):
        if env_id not in stage_dict or stage_dict[env_id].shape[0] != num_envs:
            stage_dict[env_id] = torch.zeros(num_envs, dtype=torch.bool, device=device)
        stage_dict[env_id] = torch.where(
            is_reset, torch.zeros_like(stage_dict[env_id]), stage_dict[env_id]
        )

    _stage2_reached[env_id] |= (block_z > lift_height) & (~is_reset)
    _stage3_reached[env_id] |= (block_y < y_target) & _stage2_reached[env_id]
    place_cond = block_z < place_height
    if require_ee_low:
        place_cond = place_cond & (ee_z < place_height)
    _stage4_reached[env_id] |= place_cond & _stage3_reached[env_id]

    if debug:
        if (
            env_id not in _debug_max_block_z
            or _debug_max_block_z[env_id].shape[0] != num_envs
        ):
            _debug_max_block_z[env_id] = torch.full(
                (num_envs,), float("-inf"), device=device
            )
        # clear per-episode peak on reset, then update with this step
        _debug_max_block_z[env_id] = torch.where(
            is_reset,
            torch.full_like(_debug_max_block_z[env_id], float("-inf")),
            _debug_max_block_z[env_id],
        )
        _debug_max_block_z[env_id] = torch.maximum(_debug_max_block_z[env_id], block_z)

        c = _debug_call_counter.get(env_id, 0)
        if c % max(debug_every, 1) == 0:
            s2 = _stage2_reached[env_id].sum().item()
            s3 = _stage3_reached[env_id].sum().item()
            s4 = _stage4_reached[env_id].sum().item()
            obj_w = env.scene[object_cfg.name].data.root_pos_w[:, :3]
            ee_w = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]
            robot_w = env.scene[robot_cfg.name].data.root_pos_w[:, :3]
            peak_z = _debug_max_block_z[env_id][0].item()
            print(
                f"[pick_place_sparse] call={c} envs={num_envs} "
                f"s2={s2} s3={s3} s4={s4}\n"
                f"  robot-frame : cube=({obj_b[0,0]:.3f},{obj_b[0,1]:.3f},{obj_b[0,2]:.3f}) "
                f"ee_z={ee_b[0,2]:.3f}   peak cube_z (since reset)={peak_z:.3f}\n"
                f"  world-frame : cube=({obj_w[0,0]:.3f},{obj_w[0,1]:.3f},{obj_w[0,2]:.3f}) "
                f"ee=({ee_w[0,0]:.3f},{ee_w[0,1]:.3f},{ee_w[0,2]:.3f}) "
                f"robot_base=({robot_w[0,0]:.3f},{robot_w[0,1]:.3f},{robot_w[0,2]:.3f})\n"
                f"  thresholds  : lift={lift_height} y={y_target} place={place_height}"
            )
        _debug_call_counter[env_id] = c + 1

    return torch.where(
        _stage4_reached[env_id],
        torch.full((num_envs,), reward_amount, device=device),
        torch.zeros(num_envs, device=device),
    )
