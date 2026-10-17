"""Demo path resolution and loading for Isaac Franka training (repo-root relative)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from mpail2.envs import resolve_demo_path as _resolve_demo_path
from mpail2.envs import resolve_existing_demo_path as _resolve_existing_demo_path
from mpail2.envs.gym_mujoco.utils.data_utils import load_demonstrations as _load_demo_dict


def _repo_root() -> Path:
    # mpail2/train/utils/demos.py -> parents: train, mpail2, repo root
    return Path(__file__).resolve().parent.parent.parent.parent


def resolve_demo_path(cfg_path: Optional[str], env_key: str, default_rel: str) -> str:
    """Prefer explicit config, then env var, then default; return existing path."""
    root = _repo_root()
    train_isaac = root / "mpail2" / "train" / "isaac_franka"
    search_dirs = [
        str(train_isaac),
        str(root / "mpail2" / "envs" / "isaac" / "franka"),
        str(root / "mpail2" / "envs" / "isaac" / "franka" / "demos"),
        str(train_isaac / "demos"),
    ]
    return _resolve_demo_path(
        cfg_path=cfg_path,
        env_var_key=env_key,
        default_path=default_rel,
        search_dirs=search_dirs,
    )


def load_demonstrations(path: str, device: str = "cuda"):
    root = _repo_root()
    train_isaac = root / "mpail2" / "train" / "isaac_franka"
    rel = Path(path.lstrip("./"))
    search_dirs = [
        str(train_isaac),
        str(root),
        str(root / "mpail2" / "envs" / "isaac" / "franka"),
        str(root / "mpail2" / "envs" / "isaac" / "franka" / "demos"),
        str(root / "mpail2" / "envs" / "isaac" / "franka" / rel.parent),
        str(train_isaac / "demos"),
    ]
    resolved_path = _resolve_existing_demo_path(path, search_dirs=search_dirs)
    demo, _metadata = _load_demo_dict(str(resolved_path), device=device)
    return demo
