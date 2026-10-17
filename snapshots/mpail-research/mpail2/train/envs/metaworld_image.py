"""Registry entry + Hydra ConfigStore for Meta-World image training."""

from __future__ import annotations

import mpail2.train.configs.metaworld_train_config  # noqa: F401 - ConfigStore node

ENV_NAME = "metaworld"
ENV_ALIASES: tuple[str, ...] = ("mw", "MetaWorld-MT1-image")
ENV_SPEC = {
    "suite": "metaworld_image",
    "config_name": "metaworld_image",
    "demo_env_var": "MPAIL_METAWORLD_DEMO",
    "default_demo_rel": "demos/mw_reach-v3/demos.pt",
    "default_num_iterations": 200,
}
