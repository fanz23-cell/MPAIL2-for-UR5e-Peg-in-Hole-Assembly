"""Build ``defs.LearnerConfig`` for Isaac image training.

Hydra stores only override values under ``cfg.learner`` (a plain dict). This avoids
OmegaConf schema failures on ``Type[...]`` fields while still allowing CLI overrides
like ``learner.replay_size=100000``.
"""

from __future__ import annotations

from typing import Any

import mpail2.configs.defs as defs
from omegaconf import DictConfig, OmegaConf

from mpail2.train.configs.franka_image_learner_config import (
    FrankaPickPlaceImageLearnerConfig,
    FrankaPickPlaceVisImageLearnerConfig,
    FrankaPushImageLearnerConfig,
    FrankaPushVisImageLearnerConfig,
)
from mpail2.train.configs.omnireset_learner_config import OmniResetStateLearnerConfig
from mpail2.train.utils.config_overrides import apply_dataclass_overrides, prune_placeholder_overrides


def learner_config_from_train_cfg(cfg: Any) -> defs.LearnerConfig:
    """Build task default learner and apply optional Hydra dict overrides."""
    task = getattr(cfg, "task", None) or ""
    vis_rollouts = bool(getattr(getattr(cfg, "runner", None), "vis_rollouts", False))

    if "PickPlace" in task:
        learner_cfg = FrankaPickPlaceVisImageLearnerConfig() if vis_rollouts else FrankaPickPlaceImageLearnerConfig()
    elif "Push" in task:
        learner_cfg = FrankaPushVisImageLearnerConfig() if vis_rollouts else FrankaPushImageLearnerConfig()
    elif "OmniReset" in task:
        learner_cfg = OmniResetStateLearnerConfig()
    else:
        raise ValueError(
            f"Cannot select learner preset for task {task!r}. "
            "Expected a registered Isaac-Franka*Push*/*PickPlace* task or an OmniReset task id."
        )

    raw = getattr(cfg, "learner", {})
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported cfg.learner type: {type(raw)}")
    effective = prune_placeholder_overrides(raw) or {}
    if effective:
        apply_dataclass_overrides(learner_cfg, effective, path="learner")

    return learner_cfg
