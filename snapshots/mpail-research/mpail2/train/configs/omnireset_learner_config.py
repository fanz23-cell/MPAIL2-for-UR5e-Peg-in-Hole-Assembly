"""Learner preset for OmniReset state-based manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from mpail2.configs.cfgs import ObsNormalizerCfg
import mpail2.configs.defs as defs


@dataclass
class OmniResetStatePlannerConfig(defs.PlannerConfig):
    action_dim: int = 7
    seed: int = 0
    num_elites: int = 32
    encoder_cfg: defs.MLPCoderConfig = field(
        default_factory=lambda: defs.MLPCoderConfig(
            obs_key="policy",
            input_dim=215,
        )
    )
    sampling_cfg: defs.PolicySamplingConfig = field(
        default_factory=lambda: defs.PolicySamplingConfig(
            num_rollouts=256,
        )
    )


@dataclass
class OmniResetStateLearnerConfig(defs.LearnerConfig):
    """State-based preset for OmniReset manager-based manipulation envs."""

    replay_size: int = 100_000
    replay_batch_size: int = 512
    use_terminations: bool = False
    obs_normalizer_cfg: ObsNormalizerCfg = field(
        default_factory=lambda: ObsNormalizerCfg(normalization_type="fixed")
    )
    planner_cfg: defs.PlannerConfig = field(default_factory=OmniResetStatePlannerConfig)
    # mpail3 uses per-action-dim entropy; -1.0 * 7 keeps the mpail2 total target -7.
    policy_learner_cfg: defs.PolicyLearnerConfig = field(
        default_factory=lambda: defs.PolicyLearnerConfig(target_entropy_per_action_dim=-1.0)
    )
    # With 32 envs * 160 steps = 5120 samples/iter, replay_ratio=1.0 gives 5120 disc updates/iter
    # (vs ~100 for Franka 1-env setup). Cap to ~100 updates to prevent discriminator divergence.
    replay_ratio: float = 100.0 / (32 * 160)  # ≈ 0.0195
    # WGAN-GP standard coefficient is 10.0; default 0.1 is too small to constrain Lipschitz
    reward_learner_cfg: defs.RewardLearnerConfig = field(
        default_factory=lambda: defs.RewardLearnerConfig(gp_coeff=10.0)
    )
