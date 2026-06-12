#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
DISPLAY_NAME="${DISPLAY_NAME:-${DISPLAY:-:1}}"
FFMPEG="${FFMPEG:-$HOME/miniconda3/envs/mpail2-omnireset/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"

RUN_DIR="${RUN_DIR:-$ROOT/logs/obs215_S191_headlessRender_S190config_10iter_wandb_20260607_233213}"
WANDB_RUN_ID="${WANDB_RUN_ID:-tmqchw6y}"
DEMO_PATH="${DEMO_PATH:-$ROOT/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_mpail.pt}"
CKPTS_STR="${CKPTS:-0 5 10}"
STEPS_PER_CHECKPOINT="${STEPS_PER_CHECKPOINT:-90}"
FPS="${FPS:-10}"
SLEEP="${SLEEP:-0.02}"
WANDB_STEP="${WANDB_STEP:-10}"

cd "$ROOT"

read -r -a CKPTS_ARRAY <<< "$CKPTS_STR"
run_name="$(basename "$RUN_DIR")"
out_dir="$RUN_DIR/training_process_videos"
out="$out_dir/${run_name}_ckpts_${CKPTS_STR// /_}.mp4"
ready="/tmp/${run_name}_screen_sequence_ready"
start="/tmp/${run_name}_screen_sequence_start"
done="/tmp/${run_name}_screen_sequence_done"
exit_file="/tmp/${run_name}_screen_sequence_exit"
player_log="/tmp/${run_name}_screen_sequence_player.log"
ffmpeg_log="/tmp/${run_name}_screen_sequence_ffmpeg.log"

mkdir -p "$out_dir"
rm -f "$ready" "$start" "$done" "$exit_file" "$out"

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
        if (w % 2) w -= 1
        if (h % 2) h -= 1
        printf "%dx%d+%d,%d\n", w, h, x, y
      }'
}

echo "[player] starting checkpoint sequence: $run_name ckpts=$CKPTS_STR"
"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python scripts/watch_mpail_omnireset_sequence.py \
    --checkpoint-dir "$RUN_DIR/models" \
    --checkpoints "${CKPTS_ARRAY[@]}" \
    --demo-path "$DEMO_PATH" \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    --num-envs 1 \
    --steps-per-checkpoint "$STEPS_PER_CHECKPOINT" \
    --print-every 45 \
    --sleep "$SLEEP" \
    --compact-obs-dim 215 \
    --raw-action-limit 40 40 80 30 80 15 35 \
    --reset-types ObjectAnywhereEEAnywhere \
    --disable-domain-randomization \
    --ready-file "$ready" \
    --start-file "$start" \
    --done-file "$done" \
    --exit-file "$exit_file" \
  > "$player_log" 2>&1 &
player_pid=$!

echo "[wait] waiting for Isaac ready marker: $ready"
for _ in $(seq 1 240); do
  [[ -f "$ready" ]] && break
  sleep 1
done
if [[ ! -f "$ready" ]]; then
  echo "[error] timed out waiting for $ready"
  tail -n 120 "$player_log" || true
  kill "$player_pid" 2>/dev/null || true
  exit 1
fi

sleep 3
xid="$(find_isaac_xid || true)"
if [[ -z "$xid" ]]; then
  echo "[error] could not find Isaac window; refusing to record full desktop"
  xwininfo -root -tree | head -n 120 || true
  kill "$player_pid" 2>/dev/null || true
  exit 1
fi

spec="$(window_grab_spec "$xid")"
size="${spec%%+*}"
offset="${spec#*+}"
echo "[record] xid=$xid spec=${size}+${offset} -> $out"

"$FFMPEG" -hide_banner -loglevel warning -y \
  -f x11grab -draw_mouse 0 -framerate "$FPS" \
  -video_size "$size" -i "${DISPLAY_NAME}+${offset}" \
  -vf "scale=1280:-2,format=yuv420p" \
  -an -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p \
  -movflags +faststart "$out" \
  > "$ffmpeg_log" 2>&1 &
rec_pid=$!

touch "$start"
for _ in $(seq 1 1200); do
  if [[ -f "$done" ]]; then
    break
  fi
  if ! kill -0 "$player_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done

kill -INT "$rec_pid" 2>/dev/null || true
wait "$rec_pid" 2>/dev/null || true
touch "$exit_file"
wait "$player_pid" || true

if [[ ! -s "$out" ]]; then
  echo "[error] no video output: $out"
  tail -n 120 "$player_log" || true
  tail -n 120 "$ffmpeg_log" || true
  exit 1
fi

"$FFMPEG" -hide_banner -v error -i "$out" -f null - >/dev/null
echo "[ok] wrote video $(du -h "$out" | awk '{print $1}') $out"

echo "[upload] append video to W&B run id=$WANDB_RUN_ID"
"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python scripts/upload_video_to_existing_wandb_run.py \
    --run-id "$WANDB_RUN_ID" \
    --project mpail2-omnireset \
    --video "$out" \
    --key "Training_Process/${run_name}_ckpts_${CKPTS_STR// /_}" \
    --step "$WANDB_STEP" \
    --fps "$FPS"

echo "[done] training process video recorded and uploaded: $out"
