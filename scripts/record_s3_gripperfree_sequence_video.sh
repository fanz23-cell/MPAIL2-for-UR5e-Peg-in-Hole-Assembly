#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
DEMO_PATH="${DEMO_PATH:-$ROOT/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt}"
DISPLAY_NAME="${DISPLAY_NAME:-${DISPLAY:-:1}}"

RUN_DIR="${RUN_DIR:-$ROOT/logs/obs22_gripperfree_explore5_500_20260519_234142/S3_slower_rewardlr5e-5_replay0p039_roll512_elite64_h7}"
WANDB_RUN_ID="${WANDB_RUN_ID:-3m83af27}"
RUN_NAME="${RUN_NAME:-S3_replay0p039_rewardlr5e-5_roll512_elite64_h7_500iter}"

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

out_dir="$RUN_DIR/checkpoint_videos"
out="$out_dir/${RUN_NAME}_training_process_0_to_500_gripperfree.mp4"
ready="/tmp/${RUN_NAME}_sequence_ready"
start="/tmp/${RUN_NAME}_sequence_start"
done_file="/tmp/${RUN_NAME}_sequence_done"
exit_file="/tmp/${RUN_NAME}_sequence_exit"
player_log="/tmp/${RUN_NAME}_sequence_player.log"
gst_log="/tmp/${RUN_NAME}_sequence_gst.log"

mkdir -p "$out_dir"
rm -f "$ready" "$start" "$done_file" "$exit_file" "$out"

echo "[player] starting $RUN_NAME"
"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python scripts/watch_mpail_omnireset_sequence.py \
    --checkpoint-dir "$RUN_DIR/models" \
    --checkpoints "${CKPTS[@]}" \
    --demo-path "$DEMO_PATH" \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    --num-envs 1 \
    --steps-per-checkpoint 90 \
    --print-every 45 \
    --sleep 0.02 \
    --compact-obs-dim 22 \
    --raw-action-limit 8 8 8 5 5 1.25 0.5 \
    --reset-types ObjectAnywhereEEGrasped \
    --ready-file "$ready" \
    --start-file "$start" \
    --done-file "$done_file" \
    --exit-file "$exit_file" \
    > "$player_log" 2>&1 &
player_pid=$!

echo "[wait] waiting for Isaac ready marker: $ready"
for _ in $(seq 1 360); do
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
  xwininfo -root -tree | head -n 100 || true
  kill "$player_pid" 2>/dev/null || true
  exit 1
fi
xid_dec=$((xid))
echo "[record] $RUN_NAME window xid=$xid/$xid_dec -> $out"

gst-launch-1.0 -e \
  ximagesrc display-name="$DISPLAY_NAME" xid="$xid_dec" use-damage=0 show-pointer=false \
  ! video/x-raw,framerate=10/1 \
  ! videoconvert \
  ! queue \
  ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2500 \
  ! mp4mux \
  ! filesink location="$out" \
  > "$gst_log" 2>&1 &
rec_pid=$!

touch "$start"

echo "[wait] waiting for sequence completion marker: $done_file"
for _ in $(seq 1 1800); do
  [[ -f "$done_file" ]] && break
  if ! kill -0 "$player_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ ! -f "$done_file" ]]; then
  echo "[error] player exited or timed out before sequence completed"
  tail -n 160 "$player_log" || true
  kill -INT "$rec_pid" 2>/dev/null || true
  wait "$rec_pid" 2>/dev/null || true
  exit 1
fi

kill -INT "$rec_pid" 2>/dev/null || true
wait "$rec_pid" 2>/dev/null || true
touch "$exit_file"
wait "$player_pid" || true

if [[ ! -s "$out" ]]; then
  echo "[error] no video output: $out"
  tail -n 120 "$gst_log" || true
  exit 1
fi
echo "[ok] wrote $(du -h "$out" | awk '{print $1}') $out"

echo "[upload] append gripper-free sequence video to existing wandb run id=$WANDB_RUN_ID"
"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python scripts/upload_video_to_existing_wandb_run.py \
    --run-id "$WANDB_RUN_ID" \
    --project mpail2-omnireset \
    --video "$out" \
    --key "Training_Process/${RUN_NAME}_0_to_500_gripperfree" \
    --step 500

echo "[done] S3 video recorded and uploaded: $out"
