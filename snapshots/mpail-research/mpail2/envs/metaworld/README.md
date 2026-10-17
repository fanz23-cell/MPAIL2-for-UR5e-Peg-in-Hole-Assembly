# Meta-World + MPAIL2

## Dependencies

- `metaworld` (see `pip install mpail2[metaworld]` or `pip install metaworld`)
- MuJoCo rendering: on headless Linux set e.g. `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` before running.

## Downloading demos from Hugging Face

Demos live under **`mpail2/envs/metaworld/demos/`** (per-task folders `mw_*-v3/`). Install extras, set your dataset repo id, then fetch:

```bash
pip install mpail2[metaworld]
export MPAIL_METAWORLD_HF_REPO=<org>/<dataset-repo>
mpail2-fetch-metaworld-demos
```

Or: `python -m mpail2.envs.metaworld.fetch_demos --repo-id <org>/<dataset-repo>`. For private repos, run `hf auth login` or set `HF_TOKEN`.

Equivalent CLI: `hf download <org>/<dataset-repo> --repo-type dataset --local-dir mpail2/envs/metaworld/demos --local-dir-use-symlinks False`.

## Environment stack

1. `gym.make(...)` after `import metaworld`. Current upstream `metaworld` registers ids such as `Meta-World/goal_observable` with `env_name="reach-v3"`. This repo accepts the shorthand **`mw/reach-v3`**, which is expanded to that call automatically (see `resolve_metaworld_gym_id_and_kwargs` in `env_factory.py`).
2. `RecordEpisodeStatistics`
3. `MetaWorldMPAILEnv` — dict observations + multi-camera RGB (float HWC in `[0,1]`)
4. `SyncVectorEnv` → `MetaWorldVectorTensorWrapper` (+ optional `VideoRecordingWrapper`)

Use `from mpail2.envs.metaworld.env_factory import setup_metaworld_environment` (the `metaworld` package `__init__` avoids eager imports).

If you pass `video_folder` to the factory, `render_mode` defaults to `"rgb_array"` when unset so `VideoRecordingWrapper` can call `render()`.

## Observation dict (learner)

| Key | Shape (per env) | Notes |
|-----|-----------------|--------|
| `proprioception` | `(7,)` | Hand xyz, gripper, goal xyz (MT1 39-dim layout) |
| `cam_<name>` | `(H, W, 3)` | One key per MuJoCo camera (default: `corner`, `gripperPOV`) |
| `object_state` | `(14,)` | Only if `include_object_state=True` |

## Expert / RL policy (full state)

Each `reset` / `step` adds to `info`:

- `metaworld_full_obs`: full MT1 vector, shape `(39,)` per env; after vectorization + tensor wrapper, shape `(num_envs, 39)` as `torch.float32`.

State-based experts (e.g. SB3 checkpoints trained on the raw 39-d observation) should read this tensor from `info`, not the learner dict.

## Demonstration file format (for future collection)

MPAIL expects a dict of tensors loaded by `torch.load`:

- For each observation key present in the training env’s `observation_space`, store  
  `demos[key].shape == (N, 2, *single_env_shape)`  
  where dimension `1` is `(obs_t, obs_{t+1})` along flattened transition index `N`.

Match keys and per-env shapes to `single_observation_space` from the vector env (no leading `num_envs` dimension in `*single_env_shape`).

Expert trajectories can additionally save `metaworld_full_obs` with shape `(N, 2, 39)` in the same convention if you want offline expert relabeling; that key is **not** used by `MultiCoder` unless you extend the encoder.

## Collecting demos from an SB3 state expert

Use [`data_collection.py`](data_collection.py) (requires `stable-baselines3`):

Checkpoint directory must contain **`best_model.pt`** or **`best_model.zip`** (SB3 PPO) and **`vec_normalize.pt`** or **`vec_normalize.pkl`** (VecNormalize pickle). The demo directory is created if needed; outputs are **`demos.pt`**, **`collection_summary.json`** (returns, success rate, etc.), and optionally **`videos/`**.

```bash
MUJOCO_GL=egl python -m mpail2.envs.metaworld.data_collection \
  /path/to/run/best \
  mpail2/envs/metaworld/demos/mw_reach-v3 \
  --env-id mw/reach-v3 \
  --num-episodes 20 \
  --video
```

Add `--video` to write `demo_ep_0000.mp4`, … under `mpail2/envs/metaworld/demos/mw_reach-v3/videos/` (created automatically). Use `--video-fps` (default 20).

The policy always sees **VecNormalize** 39-d observations. Saved `proprioception` pairs are the raw **7-d** MT1 vector (hand, gripper, goal). Camera keys match training (`cam_corner`, …). Expert **actions** are stored under `metadata["actions"]` with shape `[N, action_dim]` (same `N` as each demo tensor’s first dimension). Per-timestep **`info["success"]`** is stored as **`metadata["success"]`**: `torch.float32` tensor `[N, 2, 1]` aligned with observation transitions (not inside `demonstrations`, so MPAIL training is unchanged).

**`collection_summary.json`** duplicates the main stats in JSON: `episode_returns`, `mean_episode_return`, `total_return`, `episode_lengths`, `episode_success_any`, `episode_final_success`, `num_episodes_solved`, `success_rate`, paths, etc.

**Collection metadata** inside `demos.pt` is under `metadata["collection"]`: per-episode **returns**, **episode_lengths**, **episode_success_any** / **episode_final_success**, **num_episodes_solved**, **success_rate**, **total_return**, **mean_episode_return**, **total_transitions**, and optional **demo_video_files** / **video_dir** / **video_fps** when `--video` is set.

## Training MPAIL on Meta-World

Do **not** use `mpail2-train --env mw/...` — that path uses the flat MuJoCo stack. Use the Meta-World runner (same image size and camera names as when you collected demos):

```bash
MUJOCO_GL=egl python -m mpail2.train.metaworld_train \
  --demo mpail2/envs/metaworld/demos/mw_basketball-v3/demos.pt \
  --env-id mw/basketball-v3 \
  --num-envs 4 \
  --device cuda \
  --image-width 84 \
  --image-height 84 \
  --num-iterations 200
```

Or: `mpail2-train-metaworld` (same entry point). Add `--wandb` only if `WANDB_PROJECT` and `WANDB_ENTITY` are set. Add `--video` to log training videos under the run directory.

This uses `MetaworldMT1ImageLearnerConfig` + `sync_learner_dims_from_env` and `MetaWorldVectorTensorWrapper` via `setup_metaworld_environment`.

**Training videos:** Gym’s `Meta-World/goal_observable` creator does not accept `render_mode`; the factory falls back to `gym.make` without it so `VideoRecordingWrapper` still runs. If `render()` returns no frames for your install, disable `--video` or record from the multi-camera stack separately.

## Learner preset

See `mpail2.train.configs.metaworld_learner_config` (`MetaworldMT1ImageLearnerConfig`, `metaworld_mt1_image_planner_config`). Run `sync_learner_dims_from_env` after building the env so CNN `H,W` match `image_width` / `image_height`.
