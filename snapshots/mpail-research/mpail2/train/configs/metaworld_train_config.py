"""Hydra dataclass for Meta-World MT1 image + proprio MPAIL training (ConfigStore)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

from mpail2.envs.metaworld.mw_obs import DEFAULT_CAMERAS
from mpail2.train.configs.gym_train_config import GymLogHydraCfg, GymRunnerHydraCfg


@dataclass
class MetaworldHydraTrainConfig:
    """Training config for Meta-World image observations (multi-camera aware).

    ``cameras`` and image dimensions must match how demonstrations were collected.
    """

    env_id: str = "mw/reach-v3"
    num_envs: int = 1
    max_episode_length: int = 200
    device: str = "cuda"

    demo_path: Optional[str] = None
    num_demos: Optional[int] = None

    image_width: int = 84
    image_height: int = 84
    cameras: list[str] = field(default_factory=lambda: list(DEFAULT_CAMERAS))
    # None: per-camera defaults (see mw_obs.DEFAULT_FLIP_RGB_VERTICAL_BY_CAMERA).
    flip_rgb_vertical: Optional[bool] = None
    use_goal_hidden: bool = False
    include_object_state: bool = False
    # Enable image-space rollout visualisation (logs video to wandb).
    # Uses MetaworldMT1ImageVisLearnerConfig: adds a proprioception decoder and
    # projects rollout hand positions onto the corner camera image each vis step.
    vis_rollouts: bool = False

    runner: GymRunnerHydraCfg = field(default_factory=GymRunnerHydraCfg)
    # Wandb: log.no_wandb=false log.wandb_project=... log.wandb_entity=... (Hydra CLI).
    log: GymLogHydraCfg = field(default_factory=GymLogHydraCfg)

    learner: Optional[dict[str, Any]] = None
    sync_learner_dims_from_env: bool = True


ConfigStore.instance().store(name="metaworld_image", node=MetaworldHydraTrainConfig)

__all__ = ["MetaworldHydraTrainConfig"]
