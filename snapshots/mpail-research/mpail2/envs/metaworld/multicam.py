"""Off-screen multi-camera RGB rendering for Meta-World (MuJoCo) envs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import gymnasium as gym
import mujoco
import numpy as np

from mpail2.envs.metaworld.mw_obs import (
    DEFAULT_CAMERAS,
    DEFAULT_FLIP_RGB_HORIZONTAL_AFTER_VERTICAL,
    camera_obs_key,
    flip_flags_for_cameras,
)


class MetaWorldMultiCameraRenderer:
    """Renders a stack of RGB frames from named MuJoCo cameras (one env).

    Uses :class:`mujoco.Renderer` instead of Gymnasium's ``OffScreenViewer`` so that:

    - Each ``update_scene(data, ...)`` uses the **current** ``MjData`` (safe if ``MjModel``
      is reused across ``gym.make`` instances).
    - When the compiled model changes, we build a new renderer **before** freeing the old
      GL context, which avoids all-black frames on some EGL/headless stacks after
      ``free()`` + immediate recreate.
    """

    def __init__(
        self,
        *,
        width: int = 84,
        height: int = 84,
        camera_names: Sequence[str] | None = None,
        flip_rgb_vertical: bool | Mapping[str, bool] | None = None,
    ):
        self._width = int(width)
        self._height = int(height)
        self._camera_names: tuple[str, ...] = tuple(
            camera_names if camera_names is not None else DEFAULT_CAMERAS
        )
        self._flip_rgb_vertical = flip_flags_for_cameras(
            self._camera_names, flip_rgb_vertical
        )
        self._renderer: mujoco.Renderer | None = None
        self._render_shape: tuple[int, int] | None = None
        self._renderer_model: mujoco.MjModel | None = None

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self._camera_names

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return tuple(camera_obs_key(n) for n in self._camera_names)

    def _ensure_renderer(self, model: mujoco.MjModel) -> mujoco.Renderer:
        # MuJoCo caps offscreen rendering at (vis.global_.offwidth, offheight).
        # Fail loudly if the model's buffer is too small; silent clamping would
        # produce frames smaller than the declared observation_space shape.
        gw = int(model.vis.global_.offwidth)
        gh = int(model.vis.global_.offheight)
        if self._width > gw or self._height > gh:
            raise RuntimeError(
                f"Requested render size {self._width}x{self._height} exceeds model "
                f"offscreen buffer {gw}x{gh}. Increase the model's "
                f"vis.global_.offwidth / offheight or reduce image_width / image_height."
            )
        w, h = self._width, self._height
        shape = (w, h)
        # Recreate renderer when model identity changes (e.g. after env reset
        # with a new XML) — build new before closing old to avoid blank EGL frames.
        if self._renderer is None or self._render_shape != shape or self._renderer_model is not model:
            new_r = mujoco.Renderer(model, height=h, width=w)
            old_r = self._renderer
            self._renderer = new_r
            self._render_shape = shape
            self._renderer_model = model
            if old_r is not None:
                old_r.close()
        return self._renderer

    def render_stack(self, env: gym.Env) -> np.ndarray:
        """Return ``(num_cameras, H, W, 3)`` uint8 RGB.

        Accepts any wrapper around the MW env; ``env.unwrapped`` traverses the
        full Gymnasium wrapper chain to reach the base SawyerXYZEnv with
        ``.model`` / ``.data``.
        """
        unwrapped = env.unwrapped
        model = unwrapped.model
        data = unwrapped.data
        # Sync derived quantities (e.g. site/geom xpos) before rendering.
        mujoco.mj_forward(model, data)
        renderer = self._ensure_renderer(model)
        frames: list[np.ndarray] = []
        for i, name in enumerate(self._camera_names):
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id < 0:
                raise ValueError(
                    f"Camera {name!r} not found in model (check Meta-World XML names)."
                )
            renderer.update_scene(data, camera=cam_id)
            rgb = np.asarray(renderer.render(), copy=True)
            # EGL/osmesa often returns vertically flipped images for some cameras.
            if self._flip_rgb_vertical[i]:
                rgb = np.ascontiguousarray(rgb[::-1, :, :])
            # Horizontal mirror (hardcoded per camera) corrects left-right orientation.
            if name in DEFAULT_FLIP_RGB_HORIZONTAL_AFTER_VERTICAL:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            frames.append(rgb)
        return np.stack(frames, axis=0)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._render_shape = None
            self._renderer_model = None
