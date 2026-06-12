"""Fixed position reset override for OmniReset training.

This module provides a way to force all resets to use exactly the same
fixed position, ensuring demo and training distributions match perfectly.
"""

from __future__ import annotations
from typing import Any


# Fixed position for object (peg) - these are the coordinates from the first demo episode
FIXED_OBJECT_POSE = {
    "x": 0.45,      # Fixed X position
    "y": 0.10,      # Fixed Y position
    "z": 0.15,      # Fixed Z position (above table)
    "roll": 0.0,    # Fixed roll
    "pitch": 0.0,   # Fixed pitch
    "yaw": 0.0,     # Fixed yaw
}

# Fixed position for receptive object (hole) - relative to metal support
FIXED_HOLE_POSE = {
    "x": 0.50,      # Fixed X position
    "y": 0.00,      # Fixed Y position
    "z": 0.10,      # Fixed Z position
    "roll": 0.0,    # Fixed roll
    "pitch": 0.0,   # Fixed pitch
    "yaw": 0.0,     # Fixed yaw
}


def apply_fixed_position_reset_override(env_cfg: Any) -> None:
    """Override reset event to use fixed positions instead of random ranges.

    This modifies the ObjectAnywhereEEAnywhere reset event to use a single
    fixed position for both the insertive object (peg) and receptive object (hole).

    Args:
        env_cfg: The environment configuration object to modify in-place.
    """
    events = getattr(env_cfg, "events", None)
    if events is None:
        raise ValueError("env_cfg has no events attribute")

    # For Finetune/Play tasks, the reset comes from reset_from_reset_states (MultiResetManager)
    # We need to override the dataset-based reset with fixed position ranges
    reset_from_states = getattr(events, "reset_from_reset_states", None)
    if reset_from_states is not None:
        params = getattr(reset_from_states, "params", None)
        if params is not None:
            # Add fixed_position flag to params
            params["fixed_position"] = True
            params["fixed_pose"] = FIXED_OBJECT_POSE
            print(f"[INFO] Fixed position override: object at x={FIXED_OBJECT_POSE['x']}, "
                  f"y={FIXED_OBJECT_POSE['y']}, z={FIXED_OBJECT_POSE['z']}")
            return

    # Fallback: try to find direct reset events (for non-dataset-based resets)
    reset_insertive = getattr(events, "reset_insertive_object_pose", None)
    if reset_insertive is not None:
        params = getattr(reset_insertive, "params", None)
        if params is not None and "pose_range" in params:
            # Set all ranges to single fixed values
            params["pose_range"] = {
                "x": (FIXED_OBJECT_POSE["x"], FIXED_OBJECT_POSE["x"]),
                "y": (FIXED_OBJECT_POSE["y"], FIXED_OBJECT_POSE["y"]),
                "z": (FIXED_OBJECT_POSE["z"], FIXED_OBJECT_POSE["z"]),
                "roll": (FIXED_OBJECT_POSE["roll"], FIXED_OBJECT_POSE["roll"]),
                "pitch": (FIXED_OBJECT_POSE["pitch"], FIXED_OBJECT_POSE["pitch"]),
                "yaw": (FIXED_OBJECT_POSE["yaw"], FIXED_OBJECT_POSE["yaw"]),
            }
            print(f"[INFO] Fixed position override: insertive object at x={FIXED_OBJECT_POSE['x']}, "
                  f"y={FIXED_OBJECT_POSE['y']}, z={FIXED_OBJECT_POSE['z']}")
            return

    print("[WARNING] Could not find reset event to override - fixed position may not work!")

    # Override receptive object (hole) reset to fixed position
    reset_receptive = getattr(events, "reset_receptive_object_pose", None)
    if reset_receptive is not None:
        params = getattr(reset_receptive, "params", None)
        if params is not None and "pose_range" in params:
            params["pose_range"] = {
                "x": (FIXED_HOLE_POSE["x"], FIXED_HOLE_POSE["x"]),
                "y": (FIXED_HOLE_POSE["y"], FIXED_HOLE_POSE["y"]),
                "z": (FIXED_HOLE_POSE["z"], FIXED_HOLE_POSE["z"]),
                "roll": (FIXED_HOLE_POSE["roll"], FIXED_HOLE_POSE["roll"]),
                "pitch": (FIXED_HOLE_POSE["pitch"], FIXED_HOLE_POSE["pitch"]),
                "yaw": (FIXED_HOLE_POSE["yaw"], FIXED_HOLE_POSE["yaw"]),
            }
            print(f"[INFO] Fixed position override: receptive object at x={FIXED_HOLE_POSE['x']}, "
                  f"y={FIXED_HOLE_POSE['y']}, z={FIXED_HOLE_POSE['z']}")

    # Also fix end-effector position range if present
    reset_ee = getattr(events, "reset_end_effector_pose", None)
    if reset_ee is not None:
        params = getattr(reset_ee, "params", None)
        if params is not None and "pose_range_b" in params:
            # Narrow the EE range significantly around the object
            params["pose_range_b"] = {
                "x": (FIXED_OBJECT_POSE["x"] - 0.15, FIXED_OBJECT_POSE["x"] - 0.10),
                "y": (FIXED_OBJECT_POSE["y"] - 0.05, FIXED_OBJECT_POSE["y"] + 0.05),
                "z": (FIXED_OBJECT_POSE["z"] + 0.10, FIXED_OBJECT_POSE["z"] + 0.15),
                "roll": (0.0, 0.0),
                "pitch": (1.57, 1.57),  # ~90 degrees
                "yaw": (1.57, 1.57),    # ~90 degrees
            }
            print("[INFO] Fixed position override: end-effector range narrowed around object")
