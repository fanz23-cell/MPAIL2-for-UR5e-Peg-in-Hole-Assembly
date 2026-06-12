#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-}"
BASE="${BASE:-$ROOT/logs/obs22_param4_ckptvideo_500_20260518_232008}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
DEMO_PATH="${DEMO_PATH:-$ROOT/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt}"
DISPLAY_NAME="${DISPLAY_NAME:-${DISPLAY:-:1}}"
FFMPEG="${FFMPEG:-$HOME/miniconda3/envs/mpail2-omnireset/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"

cd "$ROOT"

CKPTS=(0 50 100 150 200 250 300 350 400 450 500)

find_isaac_xid() {
  xwininfo -root -tree |
    awk '
      /\("Isaac Sim/ {
        print $1
        exit
      }'
}

window_grab_spec() {
  local xid="$1"
  xwininfo -id "$xid" |
    awk '
      /Absolute upper-left X:/ { x=$4 }
      /Absolute upper-left Y:/ { y=$4 }
      /Width:/ { w=$2 }
      /Height:/ { h=$2 }
      END {
        # H.264 yuv420p needs even dimensions.
        if (w % 2) w -= 1
        if (h % 2) h -= 1
        printf "%dx%d+%d,%d\n", w, h, x, y
      }'
}

record_run() {
  local run_name="$1"
  local wandb_id="$2"
  local run_dir="$BASE/$run_name"
  local out_dir="$run_dir/checkpoint_videos"
  local out="$out_dir/${run_name}_training_process_0_to_500_playable.mp4"
  local ready="/tmp/${run_name}_playable_sequence_ready"
  local start="/tmp/${run_name}_playable_sequence_start"
  local player_log="/tmp/${run_name}_playable_sequence_player.log"
  local ffmpeg_log="/tmp/${run_name}_playable_sequence_ffmpeg.log"

  mkdir -p "$out_dir"
  rm -f "$ready" "$start" "$out"

  echo "[player] starting $run_name"
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python scripts/watch_mpail_omnireset_sequence.py \
      --checkpoint-dir "$run_dir/models" \
      --checkpoints "${CKPTS[@]}" \
      --demo-path "$DEMO_PATH" \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
      --num-envs 1 \
      --steps-per-checkpoint 90 \
      --print-every 45 \
      --sleep 0.02 \
      --compact-obs-dim 22 \
      --force-gripper-close \
      --gripper-close-value -1.0 \
      --raw-action-limit 8 8 8 5 5 1.25 0.5 \
      --reset-types ObjectAnywhereEEGrasped \
      --ready-file "$ready" \
      --start-file "$start" \
      > "$player_log" 2>&1 &
  local player_pid=$!

  echo "[wait] waiting for Isaac ready marker: $ready"
  for _ in $(seq 1 240); do
    [[ -f "$ready" ]] && break
    sleep 1
  done
  if [[ ! -f "$ready" ]]; then
    echo "[error] timed out waiting for $ready"
    tail -n 80 "$player_log" || true
    kill "$player_pid" 2>/dev/null || true
    return 1
  fi

  sleep 3
  local xid spec size offset
  xid="$(find_isaac_xid || true)"
  if [[ -z "$xid" ]]; then
    echo "[error] could not find Isaac window; refusing to record full desktop"
    xwininfo -root -tree | head -n 80 || true
    kill "$player_pid" 2>/dev/null || true
    return 1
  fi
  spec="$(window_grab_spec "$xid")"
  size="${spec%%+*}"
  offset="${spec#*+}"
  echo "[record] $run_name xid=$xid spec=${size}+${offset} -> $out"

  "$FFMPEG" -hide_banner -loglevel warning -y \
    -f x11grab -draw_mouse 0 -framerate 10 \
    -video_size "$size" -i "${DISPLAY_NAME}+${offset}" \
    -vf "scale=1280:-2,format=yuv420p" \
    -an -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p \
    -movflags +faststart "$out" \
    > "$ffmpeg_log" 2>&1 &
  local rec_pid=$!

  touch "$start"
  wait "$player_pid" || true
  sleep 2
  kill -INT "$rec_pid" 2>/dev/null || true
  wait "$rec_pid" 2>/dev/null || true

  if [[ ! -s "$out" ]]; then
    echo "[error] no video output: $out"
    tail -n 80 "$ffmpeg_log" || true
    return 1
  fi

  "$FFMPEG" -hide_banner -v error -i "$out" -f null - >/dev/null
  echo "[ok] wrote playable $(du -h "$out" | awk '{print $1}') $out"

  echo "[upload] append playable video to existing wandb run id=$wandb_id"
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python scripts/upload_video_to_existing_wandb_run.py \
      --run-id "$wandb_id" \
      --project mpail2-omnireset \
      --video "$out" \
      --key "Training_Process/${run_name}_0_to_500_playable" \
      --step 500
}

record_run "E1_replay_ratio" "ujdnj448"
record_run "E7_reward_lr1e-4" "9tjfcxwb"

echo "[done] playable sequence videos recorded and attached to matching wandb runs"
