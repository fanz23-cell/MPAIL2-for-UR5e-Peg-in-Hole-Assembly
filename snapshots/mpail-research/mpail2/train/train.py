"""Unified training entry: Isaac Lab (Hydra) or Gymnasium / MuJoCo (Hydra).

Routing uses ``--env``: values registered for Isaac (default ``push`` if omitted) run
the Isaac + Hydra pipeline; any other ``--env`` (e.g. ``Ant-v5``) composes the ``gym``
Hydra config and runs Gym training. ``--env`` is removed before Hydra parses overrides.

Use ``--continue <run>`` to resume training from a saved run directory.  ``<run>`` may
be an absolute path or a name relative to the configured log base directory.

Examples::

    python -m mpail2.train.train --env push --headless log.no_wandb=True
    python -m mpail2.train.train --env Ant-v5 learner.replay_size=500000 log.no_wandb=True
    python -m mpail2.train.train --env mw/reach-v3 demo_path=demos/mw_reach-v3/demos.pt log.no_wandb=True
    python -m mpail2.train.train --continue mpail2_isaac --headless

Requires the package installed (``pip install -e .``) or ``PYTHONPATH`` at the repo root.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import mpail2.train.envs  # noqa: F401 - register Hydra ConfigStore nodes
from mpail2.envs.isaac.launcher import parse_launcher_args
from mpail2.train.loops.isaac_image import run_isaac_image_training
from mpail2.train.registry import TrainEnvSpec, resolve_train_env


def _parse_selected_env(default_env: str = "push") -> str:
    """Parse and remove ``--env`` from ``sys.argv`` for both backends."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", type=str, default=default_env)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return args.env


def _parse_continue_arg() -> str | None:
    """Parse and remove ``--continue`` from ``sys.argv``, return run arg or None."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--continue", dest="continue_run", type=str, default=None)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    if args.continue_run is not None:
        sys.argv = [sys.argv[0]] + remaining
    return args.continue_run


def _resolve_resume(run_arg: str):
    """Resolve *run_arg* to ``(run_dir, checkpoint_path, cfg, meta)``."""
    from mpail2.train.configs._log_dir import default_log_dir

    run_dir = run_arg if os.path.isabs(run_arg) else os.path.join(default_log_dir(), run_arg)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Pass an absolute path or a name relative to {default_log_dir()}"
        )

    # Find the latest checkpoint by iteration number.
    models_dir = os.path.join(run_dir, "models")
    checkpoints = glob.glob(os.path.join(models_dir, "model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {models_dir}")

    def _ckpt_iter(p: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    checkpoint = max(checkpoints, key=_ckpt_iter)

    # Load metadata saved by the training loop.
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.json found in {run_dir}. "
            "This run may predate --continue support; re-run from scratch to generate it."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    # Load Hydra config saved at the start of training.
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.yaml found in {run_dir}.")
    raw_cfg = OmegaConf.load(cfg_path)

    # Inject the original run_name so training functions reconstruct the correct log dir.
    if meta.get("run_name"):
        OmegaConf.update(raw_cfg, "log.run_name", meta["run_name"], merge=True)

    iter_num = _ckpt_iter(checkpoint)
    print(f"[INFO] --continue: run_dir={run_dir}")
    print(f"[INFO] --continue: checkpoint={os.path.basename(checkpoint)} (iter {iter_num})")

    return run_dir, checkpoint, raw_cfg, meta, iter_num


def _compose_config(config_name: str, overrides: list[str]):
    """Compose a Hydra config with a clean GlobalHydra state."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _run_continue(run_arg: str) -> None:
    """Resume training from a saved run directory."""
    _run_dir, checkpoint, raw_cfg, meta, iter_num = _resolve_resume(run_arg)
    suite = meta.get("suite", "isaac_image")

    total_iter = meta.get("default_num_iterations")
    if total_iter is not None:
        remaining = total_iter - (iter_num + 1)
        if remaining <= 0:
            print(
                f"[INFO] --continue: checkpoint is at iter {iter_num}, "
                f"total was {total_iter}; nothing left to train."
            )
            return
        OmegaConf.update(raw_cfg, "runner.num_learning_iterations", remaining, merge=True)
        print(f"[INFO] --continue: {remaining} iterations remaining (iter {iter_num + 1} → {total_iter}).")

    if suite == "gym":
        from mpail2.train.configs.gym_train_config import GymHydraTrainConfig
        import mpail2.train.configs.gym_train_config  # noqa: F401 - register ConfigStore

        schema = OmegaConf.structured(GymHydraTrainConfig)
        cfg = OmegaConf.merge(schema, raw_cfg)

        from mpail2.train.gym_tasks.train_gym import run_gym_training

        run_gym_training(cfg, resume_from=checkpoint)
        return

    if suite == "metaworld_image":
        from mpail2.train.configs.metaworld_train_config import MetaworldHydraTrainConfig
        import mpail2.train.configs.metaworld_train_config  # noqa: F401 - register ConfigStore

        schema = OmegaConf.structured(MetaworldHydraTrainConfig)
        cfg = OmegaConf.merge(schema, raw_cfg)
        spec = TrainEnvSpec(
            config_name=meta["config_name"],
            demo_env_var=meta["demo_env_var"],
            default_demo_rel=meta["default_demo_rel"],
            default_num_iterations=meta["default_num_iterations"],
            suite=suite,
        )

        from mpail2.train.loops.metaworld_image import run_metaworld_training

        run_metaworld_training(cfg, spec, resume_from=checkpoint)
        return

    if suite == "isaac_image":
        spec = TrainEnvSpec(
            config_name=meta["config_name"],
            demo_env_var=meta["demo_env_var"],
            default_demo_rel=meta["default_demo_rel"],
            default_num_iterations=meta["default_num_iterations"],
            suite=suite,
        )
        launcher_args = parse_launcher_args("Resume MPAIL2 Isaac training from checkpoint.")
        run_isaac_image_training(raw_cfg, launcher_args, spec, resume_from=checkpoint)
        return

    raise ValueError(f"Unknown suite {suite!r} in meta.json of {run_arg!r}")


def main(default_env: str = "pick_place_6dof") -> None:
    continue_run = _parse_continue_arg()

    if continue_run is not None:
        _run_continue(continue_run)
        return

    env_token = _parse_selected_env(default_env=default_env)

    # Meta-World: ``mw/<task>`` routes to ``metaworld_image`` with ``env_id`` set (before gym fallback).
    if env_token.startswith("mw/"):
        spec = resolve_train_env("metaworld")
        cfg = _compose_config(
            config_name=spec.config_name,
            overrides=sys.argv[1:] + [f"env_id={env_token}"],
        )
        from mpail2.train.loops.metaworld_image import run_metaworld_training

        run_metaworld_training(cfg, spec)
        return

    try:
        spec = resolve_train_env(env_token)
    except ValueError:
        import mpail2.train.configs.gym_train_config  # noqa: F401 - register ``gym`` ConfigStore node

        cfg = _compose_config(
            config_name="gym",
            overrides=sys.argv[1:] + [f"env_id={env_token}"],
        )

        from mpail2.train.gym_tasks.train_gym import run_gym_training

        run_gym_training(cfg)
        return

    if spec.suite == "isaac_image":
        launcher_args = parse_launcher_args("Train MPAIL2 on Isaac Franka image tasks.")
        cfg = _compose_config(config_name=spec.config_name, overrides=sys.argv[1:])

        run_isaac_image_training(cfg, launcher_args, spec)
        return

    if spec.suite == "metaworld_image":
        cfg = _compose_config(config_name=spec.config_name, overrides=sys.argv[1:])
        from mpail2.train.loops.metaworld_image import run_metaworld_training

        run_metaworld_training(cfg, spec)
        return

    raise ValueError(
        f"Registered env {env_token!r} has unsupported training suite {spec.suite!r}. "
        "Add a dispatcher branch in mpail2.train.train.main()."
    )


if __name__ == "__main__":
    main()
