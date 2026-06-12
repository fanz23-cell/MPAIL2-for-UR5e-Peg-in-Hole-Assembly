#!/usr/bin/env python3
"""Isaac-first training entry for GUI/live screen recording.

This mirrors the viewer scripts' import order: launch Isaac Kit before importing
Hydra, torch, gym, or the MPAIL training stack. GUI Kit is sensitive to numpy
being imported too early in this environment.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


parser = argparse.ArgumentParser(description="Train MPAIL2 after launching Isaac Kit first.")
parser.add_argument("--env", type=str, default="omnireset_peg")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.headless = _env_flag("HEADLESS", False)
args_cli.enable_cameras = _env_flag("ENABLE_CAMERAS", False)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    import mpail2.train.envs  # noqa: F401
    from mpail2.train.loops.isaac_image import run_isaac_image_training_with_active_app
    from mpail2.train.registry import resolve_train_env

    spec = resolve_train_env(args_cli.env)
    if spec.suite != "isaac_image":
        raise ValueError(f"scripts/train_isaac_first.py only supports Isaac image envs, got {spec.suite!r}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None):
        cfg = compose(config_name=spec.config_name, overrides=hydra_args)

    run_isaac_image_training_with_active_app(cfg, spec)
finally:
    if simulation_app is not None:
        simulation_app.close()
