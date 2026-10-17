"""Learner presets for Meta-World MT1 image + proprioception (MultiCoder)."""

from __future__ import annotations

from dataclasses import dataclass, field

from mpail2.configs.cfgs import ImageRolloutVisConfig, ObsNormalizerCfg
import mpail2.configs.defs as defs
from mpail2.envs.metaworld.mw_obs import DEFAULT_CAMERAS, MT1_LEARNER_PROPRIO_DIM, camera_obs_key
from mpail2.train.utils.config_overrides import to_override_template


def metaworld_mt1_image_planner_config(
    *,
    image_h: int = 84,
    image_w: int = 84,
    camera_names: tuple[str, ...] = DEFAULT_CAMERAS,
    action_dim: int = 4,
    include_object_state_encoder: bool = False,
) -> defs.PlannerConfig:
    """Build :class:`~mpail2.configs.defs.PlannerConfig` matching :class:`MetaWorldMPAILEnv`.

    After building the env, run ``sync_learner_dims_from_env`` so CNN ``H``, ``W``, and proprio
    ``input_dim`` match ``image_height``, ``image_width``, and the true proprio dimension.
    """
    coder_list: list = [
        defs.MultiCoderConfig.ProprioCoderConfig(
            obs_key="proprioception",
            input_dim=7,
        ),
    ]
    for cam in camera_names:
        coder_list.append(
            defs.CNNCoderConfig(
                obs_key=camera_obs_key(cam),
                H=image_h,
                W=image_w,
                C=3,
            )
        )
    if include_object_state_encoder:
        coder_list.append(
            defs.MLPCoderConfig(
                obs_key="object_state",
                input_dim=14,
            )
        )

    return defs.PlannerConfig(
        action_dim=action_dim,
        encoder_cfg=defs.MultiCoderConfig(coder_list=coder_list),
    )


def _default_metaworld_mt1_planner() -> defs.PlannerConfig:
    return metaworld_mt1_image_planner_config()


@dataclass
class MetaworldMT1ImageLearnerConfig(defs.LearnerConfig):
    """MPAIL defaults for MT1 dict observations (proprio + ``DEFAULT_CAMERAS`` at 84×84)."""

    replay_size: int = 50_000
    use_terminations: bool = True
    obs_normalizer_cfg: ObsNormalizerCfg = field(
        default_factory=lambda: ObsNormalizerCfg(normalization_type="cam_only")
    )
    planner_cfg: defs.PlannerConfig = field(default_factory=_default_metaworld_mt1_planner)


DEFAULT_METAWORLD_LEARNER_OVERRIDE_TEMPLATE: dict[str, object] = to_override_template(
    MetaworldMT1ImageLearnerConfig()
)


# ---------------------------------------------------------------------------
# Rollout-visualisation variant
# ---------------------------------------------------------------------------

def metaworld_mt1_image_vis_planner_config(
    *,
    image_h: int = 84,
    image_w: int = 84,
    camera_names: tuple[str, ...] = DEFAULT_CAMERAS,
    action_dim: int = 4,
    include_object_state_encoder: bool = False,
) -> defs.PlannerConfig:
    """Build a :class:`~mpail2.configs.defs.PlannerConfig` with:

    - The standard MultiCoder encoder (proprio + all cameras).
    - An MLP decoder that maps latent → proprioception (hand XYZ + gripper +
      goal XYZ, 7-D).  This gives us rollout positions in world coordinates.
    - An :class:`~mpail2.configs.cfgs.ImageRolloutVisConfig` that projects
      the predicted hand positions onto the table ("corner") camera image.

    Camera intrinsics are extracted at runtime from the MuJoCo model by calling
    ``vis.setup_camera()``.  The training loop does this automatically.
    """
    base = metaworld_mt1_image_planner_config(
        image_h=image_h,
        image_w=image_w,
        camera_names=camera_names,
        action_dim=action_dim,
        include_object_state_encoder=include_object_state_encoder,
    )

    # Proprioception decoder: latent (512-D) → hand xyz + gripper + goal xyz (7-D).
    # input_dim is set to latent_dim by PlannerConfig.__post_init__.
    base.decoder_cfg = defs.MLPDecoderConfig(
        obs_key="proprioception",
        output_dim=MT1_LEARNER_PROPRIO_DIM,
    )

    # Image-space rollout vis using the corner (table-top) camera.
    base.vis_cfg = ImageRolloutVisConfig(
        vis_n_envs=1,
        vis_n_rollouts=50,
        camera_obs_key=camera_obs_key("corner"),  # "cam_corner"
        camera_name="corner",
        image_h=image_h,
        image_w=image_w,
        # Match the flip ops applied by MetaWorldMultiCameraRenderer for "corner"
        flip_vertical=True,
        flip_horizontal=True,
        hand_xyz_slice=(0, 3),  # MT1 proprio[0:3] = hand xyz
    )

    return base


def _default_metaworld_mt1_vis_planner() -> defs.PlannerConfig:
    return metaworld_mt1_image_vis_planner_config()


@dataclass
class MetaworldMT1ImageVisLearnerConfig(MetaworldMT1ImageLearnerConfig):
    """Extends the base MT1 image config with:

    - A proprioception decoder (for rollout position decoding).
    - An ``ImageRolloutVisConfig`` that renders MPPI rollouts projected onto
      the table camera image and logs a video to wandb each vis interval.
    """

    planner_cfg: defs.PlannerConfig = field(
        default_factory=_default_metaworld_mt1_vis_planner
    )


__all__ = [
    "DEFAULT_METAWORLD_LEARNER_OVERRIDE_TEMPLATE",
    "MetaworldMT1ImageLearnerConfig",
    "MetaworldMT1ImageVisLearnerConfig",
    "metaworld_mt1_image_planner_config",
    "metaworld_mt1_image_vis_planner_config",
]
