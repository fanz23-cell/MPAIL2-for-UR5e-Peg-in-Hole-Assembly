#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-}"
BASE="${BASE:-$ROOT/logs/obs22_param4_ckptvideo_500_20260518_232008}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
DEMO_PATH="${DEMO_PATH:-$ROOT/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt}"
PROJECT="${PROJECT:-mpail2-omnireset}"
VIDEO_RUN_NAME="${VIDEO_RUN_NAME:-obs22_param4_training_process_videos_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"

CKPTS=(0 50 100 250 500)
RUNS=(
  "E1_replay_ratio"
  "E7_reward_lr1e-4"
)

render_one() {
  local run_name="$1"
  local iter="$2"
  local run_dir="$BASE/$run_name"
  local ckpt="$run_dir/models/model_${iter}.pt"
  local out_dir="$run_dir/checkpoint_videos"
  local out="$out_dir/${run_name}_model_${iter}.mp4"

  mkdir -p "$out_dir"

  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] missing checkpoint: $ckpt"
    return 0
  fi

  if [[ -s "$out" ]]; then
    echo "[skip] existing video: $out"
    return 0
  fi

  echo "[render] $run_name model_$iter -> $out"
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python scripts/eval_omnireset.py \
      --checkpoint "$ckpt" \
      --demo-path "$DEMO_PATH" \
      --output "$out" \
      --num-envs 1 \
      --num-episodes 1 \
      --max-steps 100 \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
      --compact-wrapper \
      --compact-obs-dim 22 \
      --force-gripper-close \
      --gripper-close-value -1.0 \
      --raw-action-limit 8 8 8 5 5 1.25 0.5 \
      --reset-types ObjectAnywhereEEGrasped \
      --headless \
      --enable_cameras
}

for run_name in "${RUNS[@]}"; do
  for iter in "${CKPTS[@]}"; do
    render_one "$run_name" "$iter"
  done
done

echo "[sync] original offline wandb runs"
for offline_run in \
  "$BASE/E1_replay_ratio/wandb/offline-run-20260518_232029-ujdnj448" \
  "$BASE/E7_reward_lr1e-4/wandb/offline-run-20260519_031805-9tjfcxwb"; do
  if [[ -d "$offline_run" ]]; then
    "$CONDA" run --no-capture-output -n "$ENV_NAME" wandb sync "$offline_run" || true
  fi
done

echo "[upload] checkpoint videos to wandb run: $VIDEO_RUN_NAME"
"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python scripts/upload_param4_training_videos.py \
    --base "$BASE" \
    --project "$PROJECT" \
    --run-name "$VIDEO_RUN_NAME"

echo "[done] videos and wandb upload complete"
