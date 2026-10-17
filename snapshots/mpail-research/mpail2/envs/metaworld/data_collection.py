"""Roll out an SB3 + VecNormalize Meta-World expert; save MPAIL-format multimodal demos.

Expert policy sees **normalized** 39-d state. Saved ``proprioception`` is the raw 7-d vector
(hand, gripper, goal) from the same underlying state — not the normalized expert input.
Camera keys match training (``cam_<name>``, float HWC in ``[0, 1]``).
Per-timestep ``success`` from env ``info`` is stored in ``metadata['success']``
(``torch.float32``, shape ``[N, 2, 1]``, same transition alignment as observations)
so training code that only reads ``demonstrations`` is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

import numpy as np
import torch

from mpail2.envs.gym_mujoco.utils.data_utils import (
    create_transition_pairs,
    package_mpail_demos_multimodal,
    save_demonstrations,
)
from mpail2.envs.metaworld.env_factory import resolve_metaworld_gym_id_and_kwargs
from mpail2.envs.metaworld.multicam import MetaWorldMultiCameraRenderer
from mpail2.envs.metaworld.mw_obs import DEFAULT_CAMERAS, camera_obs_key, mt1_learner_proprio

# Output layout under ``--demo-dir``: fixed demo tensor name and optional video subfolder.
_DEFAULT_DEMO_FILENAME = "demos.pt"
_COLLECTION_JSON = "collection_summary.json"
_VIDEO_SUBDIR = "videos"

# Checked in order; SB3 commonly saves ``best_model.zip`` / ``vec_normalize.pkl``.
_POLICY_FILENAMES = ("best_model.pt", "best_model.zip")
_VEC_NORM_FILENAMES = ("vec_normalize.pt", "vec_normalize.pkl")


def _resolve_checkpoint_paths(checkpoint_dir: str) -> tuple[str, str]:
    """Return ``(policy_path, vec_normalize_path)`` under *checkpoint_dir*."""
    root = os.path.abspath(os.path.expanduser(checkpoint_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")

    policy_path = next(
        (os.path.join(root, n) for n in _POLICY_FILENAMES if os.path.isfile(os.path.join(root, n))),
        None,
    )
    vec_path = next(
        (os.path.join(root, n) for n in _VEC_NORM_FILENAMES if os.path.isfile(os.path.join(root, n))),
        None,
    )
    if policy_path is None:
        raise FileNotFoundError(
            f"No policy file in {root}; expected one of: {', '.join(_POLICY_FILENAMES)}"
        )
    if vec_path is None:
        raise FileNotFoundError(
            f"No VecNormalize file in {root}; expected one of: {', '.join(_VEC_NORM_FILENAMES)}"
        )
    return policy_path, vec_path


def _load_vec_normalize_obs_stats(vec_normalize_path: str) -> dict[str, Any]:
    with open(vec_normalize_path, "rb") as f:
        vn = pickle.load(f)
    return {
        "mean": np.asarray(vn.obs_rms.mean, dtype=np.float32),
        "var": np.asarray(vn.obs_rms.var, dtype=np.float32),
        "epsilon": float(vn.epsilon),
        "clip_obs": float(vn.clip_obs),
    }


def _info_success_scalar(info: dict[str, Any]) -> float:
    """Meta-World ``info['success']`` is typically ``0.0`` / ``1.0`` (float)."""
    if not info or "success" not in info:
        return 0.0
    return float(np.asarray(info["success"], dtype=np.float64).reshape(-1)[0])


def _normalize_expert_obs(obs_39: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    x = np.asarray(obs_39, dtype=np.float32).reshape(1, -1)
    y = (x - stats["mean"]) / np.sqrt(stats["var"] + stats["epsilon"])
    return np.clip(y, -stats["clip_obs"], stats["clip_obs"])


def _reset_metaworld_episode(env: Any, *, episode_seed: int) -> tuple[Any, dict[str, Any]]:
    """Reset with a new goal / object layout for this episode.

    Farama Meta-World ``goal_observable`` / ``goal_hidden`` envs set
    ``_freeze_rand_vec = True`` after ``__init__`` so every plain ``reset()`` reuses the
    same ``_last_rand_vec``. A fresh ``gym.make(..., seed=ep)`` per episode used to
    resample; with one long-lived env we must briefly unfreeze before each reset.
    """
    u = env.unwrapped
    if hasattr(u, "_freeze_rand_vec"):
        u._freeze_rand_vec = False
    try:
        return env.reset(seed=int(episode_seed))
    finally:
        if hasattr(u, "_freeze_rand_vec"):
            u._freeze_rand_vec = True


def _save_episode_demo_video(
    *,
    frames_hwc_float: list[np.ndarray],
    out_path: str,
    fps: int,
) -> None:
    """Write one MP4 from a list of float [H, W, 3] frames in [0, 1]."""
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError("Saving demo videos requires imageio. pip install imageio imageio-ffmpeg") from e

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    u8 = [(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8) for f in frames_hwc_float]
    imageio.mimsave(out_path, u8, fps=fps, macro_block_size=1)


def collect_metaworld_demonstrations(
    *,
    expert_model_path: str,
    vec_normalize_path: str,
    env_id: str,
    num_episodes: int = 10,
    seed: int = 0,
    image_width: int = 84,
    image_height: int = 84,
    camera_names: tuple[str, ...] = DEFAULT_CAMERAS,
    flip_rgb_vertical: bool | dict[str, bool] | None = None,
    max_episode_length: int = 200,
    use_goal_hidden: bool = False,
    gym_make_kwargs: dict[str, Any] | None = None,
    video_dir: str | None = None,
    video_fps: int = 20,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return MPAIL-format demo tensors and metadata (includes ``actions`` if collected).

    Observation tensors: ``[N, 2, *]`` per key. ``actions``: ``[N, action_dim]`` aligned with
    the same transition index ``N`` as the observation pairs.

    ``metadata["collection"]`` stores episode returns, lengths, success aggregates, and optional
    demo video paths. ``metadata["success"]`` holds per-transition success pairs (see module doc).
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError(
            "collect_metaworld_demonstrations requires stable-baselines3. "
            "Install with: pip install stable-baselines3"
        ) from e

    import metaworld  # noqa: F401 — register Gym ids

    import gymnasium as gym

    # Expert was trained with SB3 VecNormalize on the full 39-d obs; load its
    # running mean/var so we can feed the same normalized input during rollout.
    stats = _load_vec_normalize_obs_stats(vec_normalize_path)
    model = PPO.load(expert_model_path, device="cpu")

    gym_id, make_kw = resolve_metaworld_gym_id_and_kwargs(
        env_id,
        use_goal_hidden=use_goal_hidden,
        gym_make_kwargs=dict(gym_make_kwargs or ()),
    )

    renderer = MetaWorldMultiCameraRenderer(
        width=image_width,
        height=image_height,
        camera_names=camera_names,
        flip_rgb_vertical=flip_rgb_vertical,
    )

    cam_keys = [camera_obs_key(n) for n in camera_names]

    all_proprio: list[np.ndarray] = []
    all_imgs: dict[str, list[np.ndarray]] = {k: [] for k in cam_keys}
    all_success: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_ends: list[int] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_success_any: list[bool] = []
    episode_final_success: list[float] = []
    video_paths: list[str] = []

    # One env + one off-screen renderer for the whole run.  Creating a new
    # env per episode would allocate a new GL context each time, which breaks
    # many headless EGL stacks (first episode OK, later episodes render black).
    make_seed_kw = {**make_kw, "seed": int(seed)}
    env = gym.make(gym_id, **make_seed_kw)

    offset = 0
    try:
        for ep in range(num_episodes):
            obs, reset_info = _reset_metaworld_episode(env, episode_seed=int(seed) + ep)
            obs = np.asarray(obs, dtype=np.float64)

            # Save raw 7-d proprio (not normalized); expert normalization is
            # only applied to the 39-d vector fed to the SB3 policy.
            ep_proprio = [mt1_learner_proprio(obs)]
            ep_success = [float(_info_success_scalar(reset_info))]
            ep_imgs = {k: [] for k in cam_keys}
            rgb = renderer.render_stack(env)
            for i, k in enumerate(cam_keys):
                ep_imgs[k].append((rgb[i].astype(np.float32) / 255.0).copy())

            ep_actions: list[np.ndarray] = []
            ep_return = 0.0

            for _ in range(max_episode_length):
                # Feed VecNormalize-d 39-d obs to the expert; save raw proprio/images.
                norm = _normalize_expert_obs(obs, stats)
                action, _ = model.predict(norm, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                ep_actions.append(action.copy())

                obs, reward, term, trunc, step_info = env.step(action)
                ep_return += float(reward)
                obs = np.asarray(obs, dtype=np.float64)

                ep_proprio.append(mt1_learner_proprio(obs))
                ep_success.append(_info_success_scalar(step_info))
                rgb = renderer.render_stack(env)
                for i, k in enumerate(cam_keys):
                    ep_imgs[k].append((rgb[i].astype(np.float32) / 255.0).copy())

                if term or trunc:
                    break

            # T states → T-1 transitions. Episode boundaries are cumulative
            # state counts used by create_transition_pairs / package_mpail_demos_multimodal.
            T = len(ep_proprio)
            assert T == len(ep_actions) + 1
            assert all(len(ep_imgs[k]) == T for k in cam_keys)
            assert len(ep_success) == T

            n_steps = len(ep_actions)
            episode_returns.append(ep_return)
            episode_lengths.append(n_steps)
            ep_succ_arr = np.asarray(ep_success, dtype=np.float32)
            episode_success_any.append(bool(np.max(ep_succ_arr) >= 0.5))
            episode_final_success.append(float(ep_succ_arr[-1]))

            if video_dir is not None:
                os.makedirs(video_dir, exist_ok=True)
                tiled: list[np.ndarray] = []
                for t in range(T):
                    planes = [ep_imgs[k][t] for k in cam_keys]
                    row = np.concatenate(planes, axis=1)
                    tiled.append(row)
                vid_name = f"demo_ep_{ep:04d}.mp4"
                vid_path = os.path.join(video_dir, vid_name)
                _save_episode_demo_video(frames_hwc_float=tiled, out_path=vid_path, fps=video_fps)
                video_paths.append(vid_path)

            all_proprio.append(np.stack(ep_proprio, axis=0))
            all_success.append(ep_succ_arr)
            for k in cam_keys:
                all_imgs[k].append(np.stack(ep_imgs[k], axis=0))
            all_actions.extend(ep_actions)

            offset += T
            episode_ends.append(offset)

    finally:
        env.close()
        renderer.close()

    proprio_cat = np.concatenate(all_proprio, axis=0)
    img_cat = {k: np.concatenate(all_imgs[k], axis=0) for k in cam_keys}
    actions_np = np.stack(all_actions, axis=0)

    obs_dict: dict[str, Any] = {"proprioception": proprio_cat}
    obs_dict.update(img_cat)

    demos = package_mpail_demos_multimodal(obs_dict, episode_boundaries=episode_ends)
    n_trans = demos["proprioception"].shape[0]
    assert actions_np.shape[0] == n_trans, (actions_np.shape, n_trans)

    succ_cat = np.concatenate(all_success, axis=0)
    succ_pairs = create_transition_pairs(succ_cat, episode_boundaries=episode_ends)
    if succ_pairs.dim() == 2:
        succ_pairs = succ_pairs.unsqueeze(-1)
    assert succ_pairs.shape[0] == n_trans, (succ_pairs.shape, n_trans)

    n_solved = int(sum(episode_success_any))
    meta: dict[str, Any] = {
        "env_id": env_id,
        "gym_id": gym_id,
        "gym_make_kwargs": make_kw,
        "num_episodes": num_episodes,
        "seed": seed,
        "expert_model_path": expert_model_path,
        "vec_normalize_path": vec_normalize_path,
        "actions": torch.from_numpy(actions_np),
        # Not in ``demonstrations``: would be fed through DictDemoDataset / encoder if included there.
        "success": succ_pairs,
        "collection": {
            "episode_returns": episode_returns,
            "total_return": float(sum(episode_returns)),
            "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "episode_lengths": episode_lengths,
            "episode_success_any": episode_success_any,
            "episode_final_success": episode_final_success,
            "num_episodes_solved": n_solved,
            "success_rate": float(n_solved / num_episodes) if num_episodes else 0.0,
            "total_transitions": int(n_trans),
            "video_dir": video_dir,
            "video_fps": video_fps if video_dir else None,
            "demo_video_files": video_paths if video_paths else None,
        },
    }
    return demos, meta


def _write_collection_summary_json(path: str, meta: dict[str, Any]) -> None:
    """Write a small JSON summary for humans / dashboards (paths, returns, success)."""
    col = meta.get("collection") or {}
    succ = meta.get("success")
    success_tensor_note = None
    if succ is not None and hasattr(succ, "shape"):
        success_tensor_note = f"torch.Tensor shape {tuple(succ.shape)} stored in demos.pt metadata['success']"
    summary: dict[str, Any] = {
        "env_id": meta.get("env_id"),
        "gym_id": meta.get("gym_id"),
        "gym_make_kwargs": meta.get("gym_make_kwargs"),
        "num_episodes": meta.get("num_episodes"),
        "seed": meta.get("seed"),
        "expert_model_path": meta.get("expert_model_path"),
        "vec_normalize_path": meta.get("vec_normalize_path"),
        "total_transitions": col.get("total_transitions"),
        "episode_returns": col.get("episode_returns"),
        "mean_episode_return": col.get("mean_episode_return"),
        "total_return": col.get("total_return"),
        "episode_lengths": col.get("episode_lengths"),
        "episode_success_any": col.get("episode_success_any"),
        "episode_final_success": col.get("episode_final_success"),
        "num_episodes_solved": col.get("num_episodes_solved"),
        "success_rate": col.get("success_rate"),
        "success_tensor": success_tensor_note,
        "video_dir": col.get("video_dir"),
        "video_fps": col.get("video_fps"),
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Collect Meta-World multimodal demos from an SB3 expert. "
            "Pass the checkpoint folder (best_model.pt or .zip, vec_normalize.pt or .pkl) "
            f"and a demo output folder (writes {_DEFAULT_DEMO_FILENAME}; optional {_VIDEO_SUBDIR}/)."
        ),
        epilog=(
            "Example:\n  MUJOCO_GL=egl python -m mpail2.envs.metaworld.data_collection "
            "runs/mw_ppo/best mpail2/envs/metaworld/demos/mw_bin-picking-v3 "
            "--env-id mw/bin-picking-v3 --video\n\n"
            "Headless Linux: set MUJOCO_GL=egl (or osmesa)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "checkpoint_dir",
        help=f"Directory containing policy ({', '.join(_POLICY_FILENAMES)}) "
        f"and VecNormalize ({', '.join(_VEC_NORM_FILENAMES)})",
    )
    p.add_argument(
        "demo_dir",
        help=f"Output directory; saves {_DEFAULT_DEMO_FILENAME} here (created if missing)",
    )
    p.add_argument("--env-id", type=str, default="mw/reach-v3")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-width", type=int, default=84)
    p.add_argument("--image-height", type=int, default=84)
    p.add_argument(
        "--cameras",
        type=str,
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated MuJoCo camera names",
    )
    p.add_argument(
        "--no-flip-rgb-vertical",
        action="store_true",
        help="Disable vertical flip for every camera (default: per-camera; corner flipped, gripperPOV not)",
    )
    p.add_argument("--max-episode-length", type=int, default=200)
    p.add_argument("--use-goal-hidden", action="store_true")
    p.add_argument(
        "--video",
        action="store_true",
        help=f"Save demo_ep_*.mp4 under demo_dir/{_VIDEO_SUBDIR}/ (created if missing)",
    )
    p.add_argument("--video-fps", type=int, default=20)
    args = p.parse_args()

    policy_path, vec_path = _resolve_checkpoint_paths(args.checkpoint_dir)
    demo_root = os.path.abspath(os.path.expanduser(args.demo_dir))
    os.makedirs(demo_root, exist_ok=True)
    out_pt = os.path.join(demo_root, _DEFAULT_DEMO_FILENAME)
    vdir = os.path.join(demo_root, _VIDEO_SUBDIR) if args.video else None

    print(f"[INFO] Policy: {policy_path}")
    print(f"[INFO] VecNormalize: {vec_path}")
    print(f"[INFO] Demo output: {out_pt}")
    if vdir:
        print(f"[INFO] Videos: {vdir}/")

    cams = tuple(s.strip() for s in args.cameras.split(",") if s.strip())
    demos, meta = collect_metaworld_demonstrations(
        expert_model_path=policy_path,
        vec_normalize_path=vec_path,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        seed=args.seed,
        image_width=args.image_width,
        image_height=args.image_height,
        camera_names=cams,
        flip_rgb_vertical=False if args.no_flip_rgb_vertical else None,
        max_episode_length=args.max_episode_length,
        use_goal_hidden=args.use_goal_hidden,
        video_dir=vdir,
        video_fps=args.video_fps,
    )
    save_demonstrations(demos, out_pt, metadata=meta)
    json_path = os.path.join(demo_root, _COLLECTION_JSON)
    _write_collection_summary_json(json_path, meta)
    print(f"[INFO] Saved {out_pt}")
    print(f"[INFO] Wrote {json_path}")
    col = meta.get("collection", {})
    if col.get("episode_returns") is not None:
        print(
            f"[INFO] Expert returns: mean={col['mean_episode_return']:.4f} "
            f"total={col['total_return']:.4f} per-ep={col['episode_returns']}"
        )
    if col.get("success_rate") is not None:
        print(
            f"[INFO] Success: rate={col['success_rate']:.4f} "
            f"solved={col.get('num_episodes_solved')}/{args.num_episodes}"
        )
    if vdir:
        print(f"[INFO] Wrote {len(col.get('demo_video_files') or [])} videos under {vdir}")


if __name__ == "__main__":
    main()
