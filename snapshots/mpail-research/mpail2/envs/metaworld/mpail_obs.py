"""Wrap a Meta-World Gymnasium env into MPAIL dict observations + expert state in ``info``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mpail2.envs.metaworld.multicam import MetaWorldMultiCameraRenderer
from mpail2.envs.metaworld.mw_obs import (
    camera_obs_key,
    MT1_FULL_DIM,
    MT1_LEARNER_PROPRIO_DIM,
    MT1_OBJECT_STATE_DIM,
    mt1_learner_proprio,
    mt1_object_state,
)


class MetaWorldMPAILEnv(gym.Wrapper):
    """Dict obs: ``proprioception``, one float image key per camera; expert vector in ``info``.

    Inner env must expose MuJoCo ``model`` / ``data`` on ``unwrapped`` (standard Meta-World).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        image_width: int = 84,
        image_height: int = 84,
        camera_names: tuple[str, ...] | None = None,
        flip_rgb_vertical: bool | Mapping[str, bool] | None = None,
        include_object_state: bool = False,
    ):
        super().__init__(env)
        self._renderer = MetaWorldMultiCameraRenderer(
            width=image_width,
            height=image_height,
            camera_names=camera_names,
            flip_rgb_vertical=flip_rgb_vertical,
        )
        self._include_object_state = include_object_state
        self._image_h = int(image_height)
        self._image_w = int(image_width)

        obs_spaces: dict[str, spaces.Space] = {
            "proprioception": spaces.Box(
                -np.inf,
                np.inf,
                (MT1_LEARNER_PROPRIO_DIM,),
                dtype=np.float32,
            ),
        }
        for name in self._renderer.camera_names:
            key = camera_obs_key(name)
            obs_spaces[key] = spaces.Box(
                0.0,
                1.0,
                (self._image_h, self._image_w, 3),
                dtype=np.float32,
            )
        if include_object_state:
            obs_spaces["object_state"] = spaces.Box(
                -np.inf,
                np.inf,
                (MT1_OBJECT_STATE_DIM,),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_spaces)
        # action_space unchanged from inner (typically Box 4,)

    def _full_obs_vector(self, obs: np.ndarray) -> np.ndarray:
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if flat.shape[0] != MT1_FULL_DIM:
            raise ValueError(
                f"Expected Meta-World MT1 observation dim {MT1_FULL_DIM}, got {flat.shape[0]}"
            )
        return flat

    def _build_obs_dict(self, full_obs: np.ndarray) -> dict[str, np.ndarray]:
        # Extract learner proprio (7-d: hand xyz + gripper + goal xyz) from full 39-d state.
        proprio = mt1_learner_proprio(full_obs.reshape(1, -1))[0]
        # Render all cameras via our own mujoco.Renderer (self.env is passed, not
        # unwrapped — render_stack does unwrapped internally to reach model/data).
        rgb_stack = self._renderer.render_stack(self.env)
        out: dict[str, np.ndarray] = {
            "proprioception": proprio.astype(np.float32, copy=False),
        }
        # Normalize uint8 [0,255] → float32 [0,1] per camera.
        for i, name in enumerate(self._renderer.camera_names):
            key = camera_obs_key(name)
            out[key] = (rgb_stack[i].astype(np.float32) / 255.0).copy()
        if self._include_object_state:
            out["object_state"] = mt1_object_state(full_obs.reshape(1, -1))[0]
        return out

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        full = self._full_obs_vector(obs)
        info = dict(info)
        # Stash the full 39-d state so expert / state-based policies can read it.
        info["metaworld_full_obs"] = full.copy()
        return self._build_obs_dict(full), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        full = self._full_obs_vector(obs)
        info = dict(info)
        info["metaworld_full_obs"] = full.copy()
        return self._build_obs_dict(full), reward, terminated, truncated, info

    def render(self):
        """RGB (H, W, 3) uint8 for video logging.

        Meta-World ``gym.make`` often omits ``render_mode``, so the inner env's
        ``render()`` fails with *Unexpected mode: None*. We reuse the same
        off-screen path as observations (explicit ``rgb_array``).

        All configured cameras are tiled left-to-right (same order as
        ``camera_names`` / observation keys) so multi-view training matches the
        saved video.
        """
        rgb_stack = self._renderer.render_stack(self.env)
        # (num_cam, H, W, 3) -> (H, num_cam * W, 3)
        return np.concatenate(rgb_stack, axis=1)

    def close(self):
        self._renderer.close()
        return super().close()
