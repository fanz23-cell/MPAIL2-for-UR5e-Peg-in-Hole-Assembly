"""Dataclass learner presets for Franka image tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from mpail2.configs.cfgs import ImageRolloutVisConfig, ObsNormalizerCfg
import mpail2.configs.defs as defs
from mpail2.train.utils.config_overrides import to_override_template

# Proprioception layout for Franka tasks:
#   Push:      6-D  [ee_vel(3), ee_pos(3)];              hand_xyz_slice = (3, 6)
#   PickPlace: 8-D  [ee_vel(3), ee_pos(3), grip(1), tr(1)]; hand_xyz_slice = (3, 6)
_FRANKA_HAND_XYZ_SLICE = (3, 6)

_FRANKA_TABLE_CAM_VIS_CFG = ImageRolloutVisConfig(
    vis_n_envs=1,
    vis_n_rollouts=50,
    camera_obs_key="table_cam",
    camera_name="table_cam",
    image_h=64,
    image_w=64,
    flip_vertical=False,
    flip_horizontal=False,
    hand_xyz_slice=_FRANKA_HAND_XYZ_SLICE,
)


@dataclass
class FrankaPushPlannerConfig(defs.PlannerConfig):
    action_dim: int = 3
    seed: int = 0
    encoder_cfg: defs.MultiCoderConfig = field(
        default_factory=lambda: defs.MultiCoderConfig(
            coder_list=[
                defs.MultiCoderConfig.ProprioCoderConfig(
                    obs_key="proprioception",
                    input_dim=6,
                ),
                defs.MultiCoderConfig.MultiCNNCoderConfig(
                    obs_key="table_cam",
                    H=64,
                    W=64,
                    C=3,
                ),
                defs.MultiCoderConfig.MultiCNNCoderConfig(
                    obs_key="wrist_cam",
                    H=64,
                    W=64,
                    C=3,
                ),
            ],
        )
    )


@dataclass
class FrankaPickPlacePlannerConfig(defs.PlannerConfig):
    action_dim: int = 4
    seed: int = 0
    encoder_cfg: defs.MultiCoderConfig = field(
        default_factory=lambda: defs.MultiCoderConfig(
            coder_list=[
                defs.MultiCoderConfig.ProprioCoderConfig(
                    obs_key="proprioception",
                    input_dim=8,
                ),
                defs.MultiCoderConfig.MultiCNNCoderConfig(
                    obs_key="table_cam",
                    H=64,
                    W=64,
                    C=3,
                ),
                defs.MultiCoderConfig.MultiCNNCoderConfig(
                    obs_key="wrist_cam",
                    H=64,
                    W=64,
                    C=3,
                ),
            ],
        )
    )


@dataclass
class FrankaImageLearnerConfig(defs.LearnerConfig):
    """Declarative learner preset with nested planner/encoder dataclass config."""

    replay_size: int = 50_000
    use_terminations: bool = False
    obs_normalizer_cfg: ObsNormalizerCfg = field(
        default_factory=lambda: ObsNormalizerCfg(normalization_type="cam_only")
    )
    planner_cfg: defs.PlannerConfig | None = None


######################## TESTING ###########################

@dataclass
class _FrankaPushImageLearnerConfig(FrankaImageLearnerConfig):
    planner_cfg: FrankaPushPlannerConfig = field(
        default_factory=lambda: FrankaPushPlannerConfig(step_size=0.9)
    )


@dataclass
class FrankaPushImageLearnerConfig(FrankaImageLearnerConfig):
    planner_cfg: defs.PlannerConfig = field(default_factory=FrankaPushPlannerConfig)


@dataclass
class FrankaPickPlaceImageLearnerConfig(FrankaImageLearnerConfig):
    planner_cfg: defs.PlannerConfig = field(default_factory=FrankaPickPlacePlannerConfig)


# ---------------------------------------------------------------------------
# Rollout-visualisation variants (image-space, table camera)
# ---------------------------------------------------------------------------

def _push_vis_planner() -> defs.PlannerConfig:
    cfg = FrankaPushPlannerConfig()
    cfg.decoder_cfg = defs.MLPDecoderConfig(obs_key="proprioception", output_dim=6)
    cfg.vis_cfg = _FRANKA_TABLE_CAM_VIS_CFG
    return cfg


def _pick_place_vis_planner() -> defs.PlannerConfig:
    cfg = FrankaPickPlacePlannerConfig()
    cfg.decoder_cfg = defs.MLPDecoderConfig(obs_key="proprioception", output_dim=8)
    cfg.vis_cfg = _FRANKA_TABLE_CAM_VIS_CFG
    return cfg


@dataclass
class FrankaPushVisImageLearnerConfig(FrankaImageLearnerConfig):
    """Push preset with proprioception decoder + image-space rollout vis."""

    planner_cfg: defs.PlannerConfig = None

    def __post_init__(self):
        if self.planner_cfg is None:
            self.planner_cfg = _push_vis_planner()


@dataclass
class FrankaPickPlaceVisImageLearnerConfig(FrankaImageLearnerConfig):
    """PickPlace preset with proprioception decoder + image-space rollout vis."""

    planner_cfg: defs.PlannerConfig = None

    def __post_init__(self):
        if self.planner_cfg is None:
            self.planner_cfg = _pick_place_vis_planner()


DEFAULT_FRANKA_LEARNER_OVERRIDE_TEMPLATE: dict[str, object] = to_override_template(
    FrankaPushImageLearnerConfig()
)
