"""MPAIL-facing wrapper for OmniReset state tasks.

The UWLab policy observation is history-stacked and fairly large.  This wrapper
keeps only a small task-space state for MPAIL, and maps normalized learner
actions to raw OSC actions.  The action scale is intentionally expert-sized:
smaller "safe" limits clip the official OmniReset teacher out of the executable
action manifold.
"""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import torch
from gymnasium import spaces


EXPERT_RAW_ACTION_LIMIT = (40.0, 40.0, 80.0, 30.0, 80.0, 15.0, 35.0)
DEFAULT_RAW_ACTION_LIMIT = EXPERT_RAW_ACTION_LIMIT


LEGACY_COMPACT_OBS_DIM = 13
COMPACT_OBS_DIM = 19
COMPACT_OBS_WITH_EE_VEL_DIM = 22
RAW_POLICY_OBS_DIM = 215
PLANNER_OBS_V2_DIM = 60
PLANNER_OBS_V3_DIM = 119


def compact_omnireset_policy_obs(
    policy_obs: torch.Tensor,
    *,
    include_ee_to_hole: bool = True,
    include_ee_vel: bool = False,
) -> torch.Tensor:
    """Convert OmniReset 215-D policy obs to a compact MPAIL task state.

    Current 215-D layout from the UWLab state task is:
      - insertive asset in receptive asset frame: 5 * 6
      - previous actions:                         5 * 7
      - joint positions:                          5 * 12
      - end-effector pose:                        5 * 6
      - insertive object pose:                    5 * 6
      - receptive object pose:                    5 * 6

    We take the latest frame from the pose terms.  The compact state is:
      [
        peg_in_hole_pose_6,
        ee_to_peg_xyz_3,
        ee_to_peg_rot_3,
        ee_to_hole_xyz_3,
        ee_to_hole_rot_3,
        gripper_joint_mean_1,
        optional ee_vel_xyz_3,
      ].
    """

    if policy_obs.shape[-1] != 215:
        raise ValueError(f"Expected OmniReset policy obs dim 215, got {policy_obs.shape[-1]}")

    peg_in_hole = policy_obs[..., 24:30]
    joint_pos_latest = policy_obs[..., 113:125]
    ee_pose_prev = policy_obs[..., 143:149]
    ee_pose = policy_obs[..., 149:155]
    peg_pose = policy_obs[..., 179:185]

    ee_to_peg_xyz = peg_pose[..., :3] - ee_pose[..., :3]
    ee_to_peg_rot = peg_pose[..., 3:6] - ee_pose[..., 3:6]
    gripper_mean = joint_pos_latest[..., 6:12].mean(dim=-1, keepdim=True)
    if not include_ee_to_hole:
        parts = [peg_in_hole, ee_to_peg_xyz, ee_to_peg_rot, gripper_mean]
        if include_ee_vel:
            parts.append(ee_pose[..., :3] - ee_pose_prev[..., :3])
        return torch.cat(parts, dim=-1)

    hole_pose = policy_obs[..., 209:215]
    ee_to_hole_xyz = hole_pose[..., :3] - ee_pose[..., :3]
    ee_to_hole_rot = hole_pose[..., 3:6] - ee_pose[..., 3:6]
    parts = [peg_in_hole, ee_to_peg_xyz, ee_to_peg_rot, ee_to_hole_xyz, ee_to_hole_rot, gripper_mean]
    if include_ee_vel:
        parts.append(ee_pose[..., :3] - ee_pose_prev[..., :3])
    return torch.cat(parts, dim=-1)


def planner_v2_omnireset_policy_obs(policy_obs: torch.Tensor) -> torch.Tensor:
    """Low-dimensional task state for planner/reward/dynamics.

    Unlike the 215-D raw policy observation, this removes previous actions and
    most joint-history terms while keeping current task geometry and local
    finite-difference motion cues.
    """

    if policy_obs.shape[-1] != 215:
        raise ValueError(f"Expected OmniReset policy obs dim 215, got {policy_obs.shape[-1]}")

    peg_in_hole_prev = policy_obs[..., 18:24]
    peg_in_hole = policy_obs[..., 24:30]
    joint_pos_latest = policy_obs[..., 113:125]
    ee_pose_prev = policy_obs[..., 143:149]
    ee_pose = policy_obs[..., 149:155]
    peg_pose_prev = policy_obs[..., 173:179]
    peg_pose = policy_obs[..., 179:185]
    hole_pose = policy_obs[..., 209:215]

    ee_to_peg = peg_pose - ee_pose
    ee_to_hole = hole_pose - ee_pose
    gripper_joints = joint_pos_latest[..., 6:12]
    ee_delta = ee_pose - ee_pose_prev
    peg_delta = peg_pose - peg_pose_prev
    peg_in_hole_delta = peg_in_hole - peg_in_hole_prev

    return torch.cat(
        [
            peg_in_hole,
            peg_in_hole_delta,
            ee_pose,
            peg_pose,
            hole_pose,
            ee_to_peg,
            ee_to_hole,
            gripper_joints,
            ee_delta,
            peg_delta,
        ],
        dim=-1,
    )


def planner_v3_omnireset_policy_obs(policy_obs: torch.Tensor) -> torch.Tensor:
    """Planner state with task geometry plus control history.

    This keeps the v2 task geometry, then adds previous actions and joint
    motion cues from the 215-D OmniReset history stack. The goal is to give
    dynamics enough information to model OSC delay/tracking and closed-loop
    recovery without feeding the full raw observation into reward/planning.
    """

    if policy_obs.shape[-1] != RAW_POLICY_OBS_DIM:
        raise ValueError(f"Expected OmniReset policy obs dim 215, got {policy_obs.shape[-1]}")

    v2 = planner_v2_omnireset_policy_obs(policy_obs)
    prev_actions = policy_obs[..., 30:65]
    joint_pos_prev = policy_obs[..., 101:113]
    joint_pos = policy_obs[..., 113:125]
    joint_delta = joint_pos - joint_pos_prev
    return torch.cat([v2, prev_actions, joint_pos, joint_delta], dim=-1)


def compact_omnireset_demonstrations(
    demonstrations: Dict[str, torch.Tensor], *, include_ee_to_hole: bool = True, include_ee_vel: bool = False
) -> Dict[str, torch.Tensor]:
    """Apply the compact observation transform to loaded MPAIL demos."""

    if "policy" not in demonstrations:
        return demonstrations
    demo = demonstrations["policy"]
    if demo.shape[-1] != 215:
        return demonstrations
    if include_ee_to_hole == "planner_v2":
        return {"policy": planner_v2_omnireset_policy_obs(demo)}
    if include_ee_to_hole == "planner_v3":
        return {"policy": planner_v3_omnireset_policy_obs(demo)}
    compact = compact_omnireset_policy_obs(
        demo,
        include_ee_to_hole=include_ee_to_hole,
        include_ee_vel=include_ee_vel,
    )
    return {"policy": compact}


class OmniResetMPAILWrapper(gym.Wrapper):
    """Compact-observation and normalized-action wrapper for OmniReset."""

    def __init__(
        self,
        env: gym.Env,
        *,
        raw_action_limit: list[float] | tuple[float, ...] = DEFAULT_RAW_ACTION_LIMIT,
        compact_obs_dim: int = COMPACT_OBS_DIM,
        require_held_reset: bool = False,
        held_distance_threshold: float = 0.08,
        max_held_reset_attempts: int = 100,
        force_gripper_close: bool = False,
        gripper_close_value: float = -1.0,
        obs_key: str = "policy",
    ):
        super().__init__(env)
        self.obs_key = obs_key
        self.require_held_reset = require_held_reset
        self.held_distance_threshold = held_distance_threshold
        self.max_held_reset_attempts = max_held_reset_attempts
        self.force_gripper_close = force_gripper_close
        self.gripper_close_value = gripper_close_value
        if compact_obs_dim not in (
            LEGACY_COMPACT_OBS_DIM,
            COMPACT_OBS_DIM,
            COMPACT_OBS_WITH_EE_VEL_DIM,
            RAW_POLICY_OBS_DIM,
            PLANNER_OBS_V2_DIM,
            PLANNER_OBS_V3_DIM,
        ):
            raise ValueError(
                f"compact_obs_dim must be {LEGACY_COMPACT_OBS_DIM}, {COMPACT_OBS_DIM}, "
                f"{COMPACT_OBS_WITH_EE_VEL_DIM}, {PLANNER_OBS_V2_DIM}, {PLANNER_OBS_V3_DIM}, "
                f"or {RAW_POLICY_OBS_DIM}, got {compact_obs_dim}"
            )
        self.compact_obs_dim = compact_obs_dim
        self.include_ee_to_hole = compact_obs_dim in (COMPACT_OBS_DIM, COMPACT_OBS_WITH_EE_VEL_DIM)
        self.include_ee_vel = compact_obs_dim == COMPACT_OBS_WITH_EE_VEL_DIM
        self.raw_action_limit = torch.as_tensor(raw_action_limit, dtype=torch.float32, device=self.unwrapped.device)
        if self.raw_action_limit.numel() != 7:
            raise ValueError(f"raw_action_limit must have 7 dims, got {self.raw_action_limit.numel()}")
        self.raw_action_limit = self.raw_action_limit.view(1, 7)

        num_envs = int(getattr(self.unwrapped, "num_envs", 1))
        single_spaces = {
            obs_key: spaces.Box(low=-float("inf"), high=float("inf"), shape=(compact_obs_dim,), dtype="float32")
        }
        batched_spaces = {
            obs_key: spaces.Box(low=-float("inf"), high=float("inf"), shape=(num_envs, compact_obs_dim), dtype="float32")
        }
        if compact_obs_dim in (PLANNER_OBS_V2_DIM, PLANNER_OBS_V3_DIM):
            single_spaces["raw_policy"] = spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(RAW_POLICY_OBS_DIM,), dtype="float32"
            )
            batched_spaces["raw_policy"] = spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(num_envs, RAW_POLICY_OBS_DIM), dtype="float32"
            )
        self.single_observation_space = spaces.Dict(single_spaces)
        self.observation_space = spaces.Dict(batched_spaces)
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype="float32")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_envs, 7), dtype="float32")
        self.last_raw_policy_obs = None

    def reset(self, **kwargs: Any):
        if self.require_held_reset:
            return self._reset_until_held(**kwargs)
        obs, info = self.env.reset(**kwargs)
        self._cache_raw_policy_obs(obs)
        return self._compact_obs(obs), info

    def step(self, action):
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.raw_action_limit.device)
        normalized_action = action.clamp(-1.0, 1.0)
        if self.force_gripper_close:
            normalized_action = normalized_action.clone()
            normalized_action[..., 6] = self.gripper_close_value
        raw_action = normalized_action * self.raw_action_limit.to(
            device=normalized_action.device, dtype=normalized_action.dtype
        )
        obs, reward, terminated, truncated, info = self.env.step(raw_action)
        self._cache_raw_policy_obs(obs)
        if isinstance(info, dict):
            info["mpail_omnireset/normalized_action"] = normalized_action.detach()
            info["mpail_omnireset/raw_action"] = raw_action.detach()
        compact_obs = self._compact_obs(obs)
        if self.require_held_reset and bool((terminated | truncated).any()):
            if self.unwrapped.num_envs != 1:
                raise RuntimeError("require_held_reset currently supports num_envs=1 after episode reset.")
            compact_obs, reset_info = self._reset_until_held()
            if isinstance(info, dict):
                info["mpail_omnireset/held_reset_info"] = reset_info
        return compact_obs, reward, terminated, truncated, info

    def _cache_raw_policy_obs(self, obs):
        raw = obs[self.obs_key] if isinstance(obs, dict) else obs
        self.last_raw_policy_obs = raw.detach()

    def _compact_obs(self, obs):
        if self.compact_obs_dim == RAW_POLICY_OBS_DIM:
            raw = obs[self.obs_key] if isinstance(obs, dict) else obs
            return {self.obs_key: raw}
        if self.compact_obs_dim == PLANNER_OBS_V2_DIM:
            raw = obs[self.obs_key] if isinstance(obs, dict) else obs
            return {
                self.obs_key: planner_v2_omnireset_policy_obs(raw),
                "raw_policy": raw,
            }
        if self.compact_obs_dim == PLANNER_OBS_V3_DIM:
            raw = obs[self.obs_key] if isinstance(obs, dict) else obs
            return {
                self.obs_key: planner_v3_omnireset_policy_obs(raw),
                "raw_policy": raw,
            }
        if isinstance(obs, dict):
            compact = compact_omnireset_policy_obs(
                obs[self.obs_key],
                include_ee_to_hole=self.include_ee_to_hole,
                include_ee_vel=self.include_ee_vel,
            )
            return {self.obs_key: compact}
        return compact_omnireset_policy_obs(
            obs,
            include_ee_to_hole=self.include_ee_to_hole,
            include_ee_vel=self.include_ee_vel,
        )

    def _held_distance(self, compact_obs) -> torch.Tensor:
        policy_obs = compact_obs[self.obs_key] if isinstance(compact_obs, dict) else compact_obs
        return policy_obs[..., 6:9].norm(dim=-1)

    def _reset_until_held(self, **kwargs: Any):
        if self.unwrapped.num_envs != 1:
            raise RuntimeError("require_held_reset currently supports num_envs=1.")

        last_distance = None
        for attempt in range(1, self.max_held_reset_attempts + 1):
            obs, info = self.env.reset(**kwargs)
            compact_obs = self._compact_obs(obs)
            distance = self._held_distance(compact_obs)
            last_distance = float(distance.detach().mean().cpu().item())
            if bool((distance < self.held_distance_threshold).all()):
                if isinstance(info, dict):
                    info["mpail_omnireset/held_reset_attempts"] = attempt
                    info["mpail_omnireset/held_reset_distance"] = last_distance
                print(
                    f"[INFO] Held reset accepted after {attempt} attempt(s): "
                    f"ee_peg={last_distance:.4f}",
                    flush=True,
                )
                return compact_obs, info

        raise RuntimeError(
            "Could not find a held OmniReset reset: "
            f"threshold={self.held_distance_threshold}, attempts={self.max_held_reset_attempts}, "
            f"last_ee_peg={last_distance}"
        )
