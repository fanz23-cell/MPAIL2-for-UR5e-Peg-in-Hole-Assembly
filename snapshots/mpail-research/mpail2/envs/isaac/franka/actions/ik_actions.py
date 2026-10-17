"""IK action wrappers for Franka push tasks."""

from __future__ import annotations

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


def clip_tabletop_xyz_command(
    xyz: torch.Tensor,
    ee_pos_curr: torch.Tensor,
    workspace_lower: torch.Tensor,
    workspace_upper: torch.Tensor,
    workspace_epsilon: float,
    *,
    contact_enabled: bool,
    contact_locked: torch.Tensor,
    wrist_joint_torques: torch.Tensor,
    contact_torque_threshold: float,
    contact_unlock_threshold: float,
) -> torch.Tensor:
    """Workspace clamp + contact-torque Z-lock for normalized XYZ command (same semantics as legacy Broadcast3To6).

    ``wrist_joint_torques`` should match the slice ``applied_torque[:, 4:7]`` used for contact detection.
    ``contact_locked`` is updated in-place across steps.
    """
    clipped = xyz.clone()

    for axis in range(3):
        a = clipped[:, axis]
        mask = (
            (ee_pos_curr[:, axis] <= workspace_lower[axis] + workspace_epsilon) & (a < 0)
        ) | ((ee_pos_curr[:, axis] >= workspace_upper[axis] - workspace_epsilon) & (a > 0))
        clipped[:, axis] = torch.where(mask, torch.zeros_like(a), a)

    if contact_enabled:
        contact_detected = torch.abs(wrist_joint_torques[:, 2]) > contact_torque_threshold
        contact_locked[:] = contact_locked | contact_detected
        moving_up = clipped[:, 2] > contact_unlock_threshold
        contact_locked[:] = contact_locked & ~moving_up
        block_down = contact_locked & (clipped[:, 2] < 0)
        clipped[:, 2] = torch.where(block_down, torch.zeros_like(clipped[:, 2]), clipped[:, 2])

    return clipped


class TabletopFrankaIKActionBase(DifferentialInverseKinematicsAction):
    """Shared workspace bounds and contact Z-lock state for tabletop Franka IK actions."""

    def _init_tabletop_workspace(self, cfg: "Broadcast3To6IKActionCfg") -> None:
        self._workspace_lower = torch.tensor(cfg.workspace_lower, device=self.device, dtype=torch.float32)
        self._workspace_upper = torch.tensor(cfg.workspace_upper, device=self.device, dtype=torch.float32)
        self._workspace_epsilon = cfg.workspace_epsilon
        self._contact_torque_threshold = cfg.contact_torque_threshold
        self._contact_enabled = cfg.contact_detection_enabled
        self._contact_unlock_threshold = cfg.contact_unlock_threshold
        self._contact_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        super().reset(env_ids)
        self._contact_locked[env_ids] = False


class Broadcast3To6IKAction(TabletopFrankaIKActionBase):
    """3-DoF IK action that broadcasts XYZ commands to full 6-DoF IK (zero rotation delta).

    Includes optional workspace clamping and contact-torque-based Z-lock.
    """

    def __init__(self, cfg: "Broadcast3To6IKActionCfg", env):
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._init_tabletop_workspace(cfg)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def num_actions(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        wrist_torques = self._asset.data.applied_torque[:, 4:7]
        clipped = clip_tabletop_xyz_command(
            actions,
            ee_pos_curr,
            self._workspace_lower,
            self._workspace_upper,
            self._workspace_epsilon,
            contact_enabled=self._contact_enabled,
            contact_locked=self._contact_locked,
            wrist_joint_torques=wrist_torques,
            contact_torque_threshold=self._contact_torque_threshold,
            contact_unlock_threshold=self._contact_unlock_threshold,
        )

        self._processed_actions[:, 0:3] = clipped * self._scale[:, 0:3]
        self._processed_actions[:, 3:6] = 0.0

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        super().apply_actions()

    def get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._compute_frame_pose()


@configclass
class Broadcast3To6IKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Config for :class:`Broadcast3To6IKAction`."""

    class_type: type = Broadcast3To6IKAction

    workspace_lower: tuple[float, float, float] = (0.30, -0.15, 0.0)
    workspace_upper: tuple[float, float, float] = (0.60, 0.15, 0.26)
    workspace_epsilon: float = 1e-6
    contact_detection_enabled: bool = True
    contact_torque_threshold: float = 6.0
    contact_unlock_threshold: float = 0.1


class Full6IKAction(TabletopFrankaIKActionBase):
    """6-DoF IK (XYZ + rotation deltas); uses the same tabletop XYZ clipping as :class:`Broadcast3To6IKAction`."""

    def __init__(self, cfg: "Full6IKActionCfg", env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._init_tabletop_workspace(cfg)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def num_actions(self) -> int:
        return 6

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        wrist_torques = self._asset.data.applied_torque[:, 4:7]
        clipped = clip_tabletop_xyz_command(
            actions[:, :3],
            ee_pos_curr,
            self._workspace_lower,
            self._workspace_upper,
            self._workspace_epsilon,
            contact_enabled=self._contact_enabled,
            contact_locked=self._contact_locked,
            wrist_joint_torques=wrist_torques,
            contact_torque_threshold=self._contact_torque_threshold,
            contact_unlock_threshold=self._contact_unlock_threshold,
        )

        self._processed_actions[:, 0:3] = clipped * self._scale[:, 0:3]
        self._processed_actions[:, 3:6] = actions[:, 3:6] * self._scale[:, 3:6]

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)


@configclass
class Full6IKActionCfg(Broadcast3To6IKActionCfg):
    """Config for :class:`Full6IKAction` (unlocked orientation).

    Isaac Lab's IK term expects ``scale`` length **6** (matches pose ``action_dim``). If you pass only
    three values (position, same as :class:`Broadcast3To6IKAction`), they are appended with
    ``rotation_scale``.
    """

    class_type: type = Full6IKAction
    scale: tuple[float, ...] = (0.03, 0.03, 0.03, 0.12, 0.12, 0.12)
    rotation_scale: tuple[float, float, float] = (0.12, 0.12, 0.12)

    def __post_init__(self):
        s = self.scale
        if isinstance(s, (tuple, list)) and len(s) == 3:
            object.__setattr__(self, "scale", tuple(s) + tuple(self.rotation_scale))
