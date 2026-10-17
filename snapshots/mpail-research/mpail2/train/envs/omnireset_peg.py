"""Hydra config for OmniReset peg insertion (state) task."""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from mpail2.train.configs.isaac_base_config import IsaacHydraTrainConfig, IsaacRunnerCfg, LogConfig

ENV_NAME = "omnireset_peg"
ENV_ALIASES = (
    "omnireset-peg",
    "peg_in_hole",
    "peg-in-hole",
    "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0",
)
ENV_SPEC = {
    "suite": "isaac_image",
    "config_name": "omnireset_peg_state",
    "demo_env_var": "MPAIL_OMNIRESET_PEG_DEMO",
    "default_demo_rel": "../demos/omnireset_peg_state.pt",
    "default_num_iterations": 500,
    "import_modules": ("uwlab_tasks",),
}


@dataclass
class OmniResetPegTrainConfig(IsaacHydraTrainConfig):
    task: str = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0"
    num_envs: int = 32
    omnireset_mpail_wrapper: bool = True
    omnireset_raw_action_limit: list[float] = field(
        default_factory=lambda: [40.0, 40.0, 80.0, 30.0, 80.0, 15.0, 35.0]
    )
    log: LogConfig = field(
        default_factory=lambda: LogConfig(
            run_name="mpail2_omnireset_peg",
        )
    )
    runner: IsaacRunnerCfg = field(
        default_factory=lambda: IsaacRunnerCfg(num_learning_iterations=500),
    )


ConfigStore.instance().store(name="omnireset_peg_state", node=OmniResetPegTrainConfig)
