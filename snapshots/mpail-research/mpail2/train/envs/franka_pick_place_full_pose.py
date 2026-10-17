"""Hydra config for Franka pick-and-place (image, 6-DoF EE + gripper) — registers ConfigStore node."""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from mpail2.train.configs.isaac_base_config import IsaacHydraTrainConfig, IsaacRunnerCfg, LogConfig

ENV_NAME = "pick_place_6dof"
ENV_ALIASES = (
    "pick-place-full-pose",
    "pickplace-full-pose",
    "Isaac-FrankaPickPlace-Image-Dense-FullPose-v0",
)
ENV_SPEC = {
    "suite": "isaac_image",
    "config_name": "pick_place_image_full_pose",
    "demo_env_var": "MPAIL_PICK_PLACE_DEMO",
    "default_demo_rel": "../demos/pick_place_image.pt",
    "default_num_iterations": 200,
}


@dataclass
class FrankaPickPlaceFullPoseTrainConfig(IsaacHydraTrainConfig):
    task: str = "Isaac-FrankaPickPlace-Image-Dense-FullPose-v0"
    log: LogConfig = field(
        default_factory=lambda: LogConfig(
            run_name="mpail2_pick_place_full_pose",
            video_length=200,
        )
    )
    runner: IsaacRunnerCfg = field(
        default_factory=lambda: IsaacRunnerCfg(num_learning_iterations=200),
    )


ConfigStore.instance().store(name="pick_place_image_full_pose", node=FrankaPickPlaceFullPoseTrainConfig)
