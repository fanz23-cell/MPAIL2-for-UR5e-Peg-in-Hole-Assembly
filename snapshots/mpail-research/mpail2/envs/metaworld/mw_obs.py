"""Meta-World MT1 observation layout helpers (39-dim state) and camera naming.

Layout matches ``metaworld.sawyer_xyz_env.SawyerXYZEnv._get_obs``:

- ``[0:18]`` — current frame: hand xyz (3), gripper (1), object block (14)
- ``[18:36]`` — previous frame (same structure)
- ``[36:39]`` — goal position (xyz)

Learner proprio (no object pose) uses hand, gripper, and goal from the current block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# Default MuJoCo camera names in Meta-World XMLs (fixed + gripper POV).
DEFAULT_CAMERAS: tuple[str, ...] = ("corner", "gripperPOV")

# Per-camera vertical RGB flip after MuJoCo render (EGL corner view is often inverted).
# gripper POV is typically already upright.
DEFAULT_FLIP_RGB_VERTICAL_BY_CAMERA: dict[str, bool] = {
    "corner": True,
    "gripperPOV": False,
}

# After the vertical flip above, mirror left–right for these cameras (corner view).
DEFAULT_FLIP_RGB_HORIZONTAL_AFTER_VERTICAL: frozenset[str] = frozenset({"corner"})


def flip_flags_for_cameras(
    camera_names: Sequence[str],
    flip_rgb_vertical: bool | Mapping[str, bool] | None,
) -> tuple[bool, ...]:
    """Resolve vertical flip flags for ``camera_names`` (order preserved).

    - ``None``: use :data:`DEFAULT_FLIP_RGB_VERTICAL_BY_CAMERA` per name; unknown cameras default
      to ``True`` (match legacy global flip).
    - ``bool``: apply the same flag to every camera.
    - ``Mapping``: explicit per-camera; names omitted use :data:`DEFAULT_FLIP_RGB_VERTICAL_BY_CAMERA`
      then ``True``.
    """
    names = tuple(camera_names)
    if flip_rgb_vertical is None:
        return tuple(DEFAULT_FLIP_RGB_VERTICAL_BY_CAMERA.get(n, True) for n in names)
    if isinstance(flip_rgb_vertical, bool):
        return (flip_rgb_vertical,) * len(names)
    m: Mapping[str, bool] = flip_rgb_vertical
    return tuple(
        m[n] if n in m else DEFAULT_FLIP_RGB_VERTICAL_BY_CAMERA.get(n, True) for n in names
    )


def camera_obs_key(camera_name: str) -> str:
    """Stable observation dict key for a MuJoCo camera name."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in camera_name)
    return f"cam_{safe}"


HAND_POS_SLICE = slice(0, 3)
GRIPPER_SLICE = slice(3, 4)
CURRENT_OBJECT_SLICE = slice(4, 18)
GOAL_SLICE = slice(36, 39)

MT1_FULL_DIM = 39
MT1_LEARNER_PROPRIO_DIM = 7
MT1_OBJECT_STATE_DIM = 14


def mt1_learner_proprio(obs: np.ndarray) -> np.ndarray:
    """Concatenate hand xyz, gripper scalar, goal xyz; shape (..., 7).

    ``obs`` must be the full MT1 vector (last axis size 39).
    """
    if obs.shape[-1] != MT1_FULL_DIM:
        raise ValueError(
            f"Expected MT1 obs dim {MT1_FULL_DIM}, got shape {obs.shape}. "
            "Adjust slices if using a different Meta-World observation layout."
        )
    return np.concatenate(
        [
            np.asarray(obs[..., HAND_POS_SLICE], dtype=np.float32),
            np.asarray(obs[..., GRIPPER_SLICE], dtype=np.float32),
            np.asarray(obs[..., GOAL_SLICE], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def mt1_object_state(obs: np.ndarray) -> np.ndarray:
    """Object-related block from the current frame; shape (..., 14)."""
    if obs.shape[-1] != MT1_FULL_DIM:
        raise ValueError(f"Expected MT1 obs dim {MT1_FULL_DIM}, got shape {obs.shape}")
    return np.asarray(obs[..., CURRENT_OBJECT_SLICE], dtype=np.float32)
