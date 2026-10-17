"""Resolve the default log directory.

Priority:
  1. ``MPAIL2_LOGS_DIR`` environment variable
  2. ``<project-root>/logs``  (mpail-research/logs)
"""

import os
from pathlib import Path

# This file lives at mpail2/train/configs/_log_dir.py → parents[3] = mpail-research/
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


def default_log_dir(subdir: str = "") -> str:
    """Return the resolved log base dir, optionally extended by *subdir*."""
    base = os.environ.get("MPAIL2_LOGS_DIR") or str(_PROJECT_ROOT / "logs")
    return f"{base}/{subdir}" if subdir else base
