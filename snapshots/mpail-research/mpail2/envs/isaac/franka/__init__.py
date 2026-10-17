import gymnasium as gym

from .pick_place_env_cfg import FrankaPickPlaceEnvCfg  # noqa: F401
from .push_env_cfg import FrankaPushEnvCfg  # noqa: F401
from .variants.image_full_pose_env_cfg import (  # noqa: F401
    FrankaPickPlaceFullPoseImageEnvCfg,
    FrankaPushFullPoseImageEnvCfg,
)
from .variants.image_pick_place_env_cfg import FrankaPickPlaceImageEnvCfg  # noqa: F401
from .variants.image_push_env_cfg import FrankaPushImageEnvCfg  # noqa: F401

##
# Register environments in the gym namespace
##

gym.register(
    id="Isaac-FrankaPush-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": FrankaPushEnvCfg},
)

gym.register(
    id="Isaac-FrankaPickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": FrankaPickPlaceEnvCfg},
)

gym.register(
    id="Isaac-FrankaPush-Image-Dense-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPushImageEnvCfg,
    },
)

gym.register(
    id="Isaac-FrankaPickPlace-Image-Dense-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": FrankaPickPlaceImageEnvCfg},
)

gym.register(
    id="Isaac-FrankaPush-Image-Dense-FullPose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": FrankaPushFullPoseImageEnvCfg},
)

gym.register(
    id="Isaac-FrankaPickPlace-Image-Dense-FullPose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": FrankaPickPlaceFullPoseImageEnvCfg},
)
