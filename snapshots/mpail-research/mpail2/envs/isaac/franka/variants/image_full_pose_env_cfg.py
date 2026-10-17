"""Image Franka tasks with 6-DoF end-effector control (orientation unlocked)."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from ..actions.ik_actions import Full6IKActionCfg
from .image_pick_place_env_cfg import FrankaPickPlaceImageEnvCfg
from .image_push_env_cfg import FrankaPushImageEnvCfg


@configclass
class FrankaPushFullPoseImageEnvCfg(FrankaPushImageEnvCfg):
    """Same as dense image push, but 6-DoF IK (XYZ + rotation) instead of translation-only."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = Full6IKActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=(0.03, 0.03, 0.03, 0.12, 0.12, 0.12),
            body_offset=Full6IKActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaPickPlaceFullPoseImageEnvCfg(FrankaPickPlaceImageEnvCfg):
    """Same as dense image pick-and-place, but 7-D actions (6-DoF IK + gripper)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = Full6IKActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=(0.03, 0.03, 0.03, 0.12, 0.12, 0.12),
            body_offset=Full6IKActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
