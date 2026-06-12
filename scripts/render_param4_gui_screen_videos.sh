#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-}"
BASE="${BASE:-$ROOT/logs/obs22_param4_ckptvideo_500_20260518_232008}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
DEMO_PATH="${DEMO_PATH:-$ROOT/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt}"
DISPLAY_NAME="${DISPLAY_NAME:-${DISPLAY:-:1}}"

cd "$ROOT"

# These are checkpoint snapshots, not final-only videos.
CKPTS=(0 50 250 500)
RUNS=(
  "E1_replay_ratio"
  "E7_reward_lr1e-4"
)

record_checkpoint() {
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
  rm -f "$out"

  echo "[screen-record] $run_name model_$iter -> $out"
  gst-launch-1.0 -e \
    ximagesrc display-name="$DISPLAY_NAME" use-damage=0 show-pointer=false \
    ! video/x-raw,framerate=10/1 \
    ! videoconvert \
    ! queue \
    ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2500 \
    ! mp4mux \
    ! filesink location="$out" \
    >/tmp/param4_gst_${run_name}_${iter}.log 2>&1 &
  local rec_pid=$!

  # Give the recorder a moment before opening Isaac.
  sleep 2
  set +e
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python scripts/watch_mpail_omnireset.py \
      --checkpoint "$ckpt" \
      --demo-path "$DEMO_PATH" \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
      --num-envs 1 \
      --num-steps 120 \
      --print-every 30 \
      --sleep 0.02 \
      --compact-wrapper \
      --compact-obs-dim 22 \
      --force-gripper-close \
      --gripper-close-value -1.0 \
      --raw-action-limit 8 8 8 5 5 1.25 0.5 \
      --reset-types ObjectAnywhereEEGrasped
  local watch_status=$?
  set -e

  sleep 2
  kill -INT "$rec_pid" 2>/dev/null || true
  wait "$rec_pid" 2>/dev/null || true

  if [[ ! -s "$out" ]]; then
    echo "[error] video was not created: $out"
    return 1
  fi
  echo "[ok] wrote $(du -h "$out" | awk '{print $1}') $out"
  return "$watch_status"
}

for run_name in "${RUNS[@]}"; do
  for iter in "${CKPTS[@]}"; do
    record_checkpoint "$run_name" "$iter" || true
  done
done

echo "[done] GUI screen checkpoint videos under $BASE/*/checkpoint_videos"
