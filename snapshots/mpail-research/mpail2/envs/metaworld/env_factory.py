"""Build vectorized Meta-World + MPAIL tensor wrappers (optional video).

Wrapping order (inner → outer):
  gym.make(Meta-World/goal_observable)   raw MuJoCo env, 39-d flat obs
  → RecordEpisodeStatistics              episode return/length tracking
  → MetaWorldMPAILEnv                    dict obs (proprio + multi-cam RGB), full state in info
  → SyncVectorEnv                        batches N sub-envs
  → MetaWorldVectorTensorWrapper         numpy→torch, preserves Dict obs keys
  → VideoRecordingWrapper (optional)     records first sub-env frames to MP4
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import gymnasium as gym

from mpail2.envs.gym_mujoco.wrappers import VideoRecordingWrapper
from mpail2.envs.metaworld.mpail_obs import MetaWorldMPAILEnv
from mpail2.envs.metaworld.mw_obs import DEFAULT_CAMERAS, flip_flags_for_cameras
from mpail2.envs.metaworld.tensor_wrapper import MetaWorldVectorTensorWrapper


def _require_metaworld() -> None:
    try:
        import metaworld  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Meta-World environments require the `metaworld` package. "
            "Install with: pip install metaworld\n"
            "Or: pip install mpail2[metaworld]"
        ) from e


def resolve_metaworld_gym_id_and_kwargs(
    env_id: str,
    *,
    use_goal_hidden: bool = False,
    gym_make_kwargs: dict | None = None,
) -> tuple[str, dict[str, Any]]:
    """Map legacy ``mw/<task>`` ids to current Gymnasium Meta-World registrations.

    Current ``metaworld`` uses ids such as ``Meta-World/goal_observable`` with
    ``env_name="reach-v3"``. The ``mw/reach-v3`` shorthand sets that up
    automatically.

    If ``env_id`` does not start with ``mw/``, it is passed through as the Gym
    id; merge ``gym_make_kwargs`` (e.g. ``env_name`` for goal_observable).
    """
    extra = dict(gym_make_kwargs or ())
    if env_id.startswith("mw/"):
        task = env_id[3:].strip()
        gid = (
            "Meta-World/goal_hidden"
            if use_goal_hidden
            else "Meta-World/goal_observable"
        )
        base = {"env_name": task}
    else:
        gid = env_id
        base = {}
    merged = {**base, **extra}
    return gid, merged


def make_metaworld_thunk(
    gym_id: str,
    gym_make_base: dict[str, Any],
    idx: int,
    base_seed: int | None,
    max_episode_length: int,
    device: str,
    image_width: int,
    image_height: int,
    camera_names: tuple[str, ...],
    flip_rgb_vertical: bool | Mapping[str, bool] | None,
    include_object_state: bool,
    render_mode: str | None,
):

    def thunk():
        _require_metaworld()
        # Per-env deterministic seed: base + sub-env index.
        seed = (0 if base_seed is None else int(base_seed)) + idx
        # max_episode_steps overrides Meta-World's default TimeLimit (500) so
        # truncation matches the runner's rollout length.
        kw = {**gym_make_base, "seed": seed, "max_episode_steps": max_episode_length}
        # Meta-World gym registrations may not accept render_mode; try with it
        # first so SyncVectorEnv.render() works, fall back gracefully.
        if render_mode is not None:
            try:
                env = gym.make(gym_id, render_mode=render_mode, **kw)
            except TypeError:
                env = gym.make(gym_id, **kw)
        else:
            env = gym.make(gym_id, **kw)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = MetaWorldMPAILEnv(
            env,
            image_width=image_width,
            image_height=image_height,
            camera_names=camera_names,
            flip_rgb_vertical=flip_rgb_vertical,
            include_object_state=include_object_state,
        )
        return env

    return thunk


@dataclass
class MetaworldEnvArgs:
    """Arguments for :func:`make_metaworld_env` / :func:`setup_metaworld_environment`.

    ``env_id`` may be a shorthand ``mw/<task>`` (e.g. ``mw/reach-v3``), which maps to
    ``Meta-World/goal_observable`` and ``env_name=<task>``. Otherwise it is the Gymnasium
    id passed to ``gym.make`` (use ``gym_make_kwargs`` for ``env_name``, etc.).

    Sub-environment ``i`` uses ``seed + i`` when calling ``gym.make``.

    ``flip_rgb_vertical``: ``None`` uses per-camera defaults in
    ``DEFAULT_FLIP_RGB_VERTICAL_BY_CAMERA`` (``corner`` flipped, ``gripperPOV`` not). A ``bool``
    applies to every camera; a mapping sets explicit flags per MuJoCo camera name.
    """

    env_id: str
    num_envs: int = 1
    max_episode_length: int = 200
    device: str = "cuda"
    seed: Optional[int] = 0
    use_goal_hidden: bool = False
    gym_make_kwargs: dict[str, Any] = field(default_factory=dict)
    image_width: int = 84
    image_height: int = 84
    camera_names: tuple[str, ...] = DEFAULT_CAMERAS
    flip_rgb_vertical: bool | Mapping[str, bool] | None = None
    include_object_state: bool = False
    render_mode: Optional[str] = None
    video_folder: Optional[str] = None
    video_step_trigger: Optional[Callable[[int], bool]] = None
    video_length: int = 200
    enable_wandb: bool = False
    video_fps: int = 30


def setup_metaworld_environment(
    env_id: str,
    num_envs: int = 1,
    max_episode_length: int = 200,
    device: str = "cuda",
    seed: Optional[int] = 0,
    use_goal_hidden: bool = False,
    gym_make_kwargs: Optional[dict[str, Any]] = None,
    image_width: int = 84,
    image_height: int = 84,
    camera_names: tuple[str, ...] = DEFAULT_CAMERAS,
    flip_rgb_vertical: bool | Mapping[str, bool] | None = None,
    include_object_state: bool = False,
    render_mode: Optional[str] = None,
    video_folder: Optional[str] = None,
    video_step_trigger: Optional[Callable[[int], bool]] = None,
    video_length: int = 200,
    enable_wandb: bool = False,
    video_fps: int = 30,
):
    """Convenience wrapper around :func:`make_metaworld_env` with positional args."""
    return make_metaworld_env(
        MetaworldEnvArgs(
            env_id=env_id,
            num_envs=num_envs,
            max_episode_length=max_episode_length,
            device=device,
            seed=seed,
            use_goal_hidden=use_goal_hidden,
            gym_make_kwargs=dict(gym_make_kwargs or ()),
            image_width=image_width,
            image_height=image_height,
            camera_names=camera_names,
            flip_rgb_vertical=flip_rgb_vertical,
            include_object_state=include_object_state,
            video_folder=video_folder,
            video_step_trigger=video_step_trigger,
            video_length=video_length,
            enable_wandb=enable_wandb,
            video_fps=video_fps,
            render_mode=render_mode,
        )
    )


def make_metaworld_env(args: MetaworldEnvArgs | Any):
    """Create ``SyncVectorEnv`` + :class:`MetaWorldVectorTensorWrapper` [+ optional video]."""
    _require_metaworld()

    env_id = args.env_id
    num_envs = args.num_envs
    max_episode_length = args.max_episode_length
    device = args.device
    image_width = args.image_width
    image_height = args.image_height
    camera_names = tuple(args.camera_names)
    flip_rgb_vertical = args.flip_rgb_vertical
    include_object_state = args.include_object_state
    video_folder = args.video_folder
    video_step_trigger = args.video_step_trigger
    video_length = args.video_length
    enable_wandb = args.enable_wandb
    video_fps = args.video_fps
    # Default to rgb_array when video is requested so render() produces frames.
    render_mode = getattr(args, "render_mode", None)
    if video_folder is not None and render_mode is None:
        render_mode = "rgb_array"

    use_goal_hidden = bool(getattr(args, "use_goal_hidden", False))
    gym_make_extra = dict(getattr(args, "gym_make_kwargs", None) or ())
    base_seed = getattr(args, "seed", None)
    gym_id, gym_make_base = resolve_metaworld_gym_id_and_kwargs(
        env_id,
        use_goal_hidden=use_goal_hidden,
        gym_make_kwargs=gym_make_extra,
    )

    # Resolve per-camera flip flags: None → defaults (corner:True, gripperPOV:False).
    flip_by_cam = dict(zip(camera_names, flip_flags_for_cameras(camera_names, flip_rgb_vertical)))
    print(
        f"[INFO] Creating {num_envs} Meta-World env(s): id={gym_id!r} kwargs={gym_make_base!r} "
        f"images={image_width}x{image_height} cameras={camera_names} flip_rgb_vertical={flip_by_cam}"
    )

    env_fns = [
        make_metaworld_thunk(
            gym_id,
            gym_make_base,
            i,
            base_seed,
            max_episode_length,
            device,
            image_width,
            image_height,
            camera_names,
            flip_rgb_vertical,
            include_object_state,
            render_mode,
        )
        for i in range(num_envs)
    ]
    # Gymnasium >=1.0 SyncVectorEnv infers render_mode from sub-envs (no kwarg).
    # Older versions (<0.29) accepted render_mode as a kwarg. Try with it first,
    # fall back to plain construction.
    if render_mode is not None:
        try:
            env = gym.vector.SyncVectorEnv(env_fns, render_mode=render_mode)
        except TypeError:
            env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.SyncVectorEnv(env_fns)

    env = MetaWorldVectorTensorWrapper(
        env,
        num_envs=num_envs,
        max_episode_length=max_episode_length,
        device=device,
    )

    if video_folder is not None:
        os.makedirs(video_folder, exist_ok=True)
        env = VideoRecordingWrapper(
            env,
            video_folder=video_folder,
            step_trigger=video_step_trigger,
            video_length=video_length,
            enable_wandb=enable_wandb,
            fps=video_fps,
        )
        print(f"[INFO] Video recording enabled: {video_folder}")

    return env


__all__ = [
    "MetaworldEnvArgs",
    "make_metaworld_env",
    "make_metaworld_thunk",
    "resolve_metaworld_gym_id_and_kwargs",
    "setup_metaworld_environment",
]
