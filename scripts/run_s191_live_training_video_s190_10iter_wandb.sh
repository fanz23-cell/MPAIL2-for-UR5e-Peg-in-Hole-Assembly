#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
DISPLAY_NAME="${DISPLAY_NAME:-${DISPLAY:-:1}}"
FFMPEG="${FFMPEG:-$HOME/miniconda3/envs/mpail2-omnireset/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"

NUM_ITERS="${NUM_ITERS:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
NUM_ENVS_VALUE="${NUM_ENVS_VALUE:-32}"
IQLEARN_PRETRAIN_STEPS="${IQLEARN_PRETRAIN_STEPS:-10000}"
IQLEARN_REWARD_DISTILL_STEPS="${IQLEARN_REWARD_DISTILL_STEPS:-2000}"
FPS="${FPS:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-mpail2-omnireset}"
WANDB_ENTITY="${WANDB_ENTITY:-fanz23-university-of-washington}"

cd "$ROOT"

action_limit='[40,40,80,30,80,15,35]'
run_id="${RUN_ID:-obs215_S191_liveTrainingVideo_S190config_${NUM_ITERS}iter_wandb_$(date +%Y%m%d_%H%M%S)}"
run_dir="$ROOT/logs/$run_id"
video_dir="$run_dir/training_process_videos"
out="$video_dir/${run_id}_live_training.mp4"
train_log="$run_dir/train.log"
ffmpeg_log="$run_dir/live_training_ffmpeg.log"

mkdir -p "$video_dir"
echo "$run_id" > "$run_dir/RUN_ID.txt"

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

cleanup() {
  if [[ -n "${rec_pid:-}" ]]; then
    kill -INT "$rec_pid" 2>/dev/null || true
    wait "$rec_pid" 2>/dev/null || true
  fi
  if [[ -n "${train_pid:-}" ]] && kill -0 "$train_pid" 2>/dev/null; then
    kill "$train_pid" 2>/dev/null || true
    wait "$train_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[train] starting live GUI training run: $run_id"
HEADLESS=0 \
LIVE_RENDER=1 \
LIVE_RENDER_INTERVAL_STEPS=1 \
LOG_VIDEO=False \
ENABLE_CAMERAS=False \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_ENTITY="$WANDB_ENTITY" \
WANDB_MODE=online \
WANDB_RUN_ID="$run_id" \
WANDB_NAME="$run_id" \
NUM_ENVS="$NUM_ENVS_VALUE" \
DEMO_PATH="$ROOT/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_mpail.pt" \
LOG_FILE="$train_log" \
TRAIN_ENTRY=scripts/train_isaac_first.py \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations="$NUM_ITERS" \
  runner.teacher_policy_path="$ROOT/mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt" \
  runner.teacher_bootstrap_steps=0 \
  runner.teacher_blend_steps=0 \
  runner.teacher_diag_always=true \
  log.run_name="$run_id" \
  log.run_log_dir="$run_dir" \
  log.checkpoint_every="$CHECKPOINT_EVERY" \
  log.no_wandb=False \
  log.video=False \
  enable_cameras=False \
  omnireset_mpail_wrapper=True \
  omnireset_compact_obs_dim=215 \
  omnireset_raw_action_limit="$action_limit" \
  +omnireset_disable_domain_randomization=True \
  omnireset_reset_types=[ObjectAnywhereEEAnywhere] \
  omnireset_reset_probs=[1.0] \
  +learner.iqlearn_pretrain_path="$ROOT/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_bc_raw.pt" \
  +learner.iqlearn_pretrain_steps="$IQLEARN_PRETRAIN_STEPS" \
  +learner.iqlearn_pretrain_batch_size=512 \
  +learner.iqlearn_alpha=0.5 \
  +learner.iqlearn_init_temp=0.01 \
  +learner.iqlearn_learn_temp=False \
  +learner.iqlearn_policy_batch_ratio=1.0 \
  +learner.iqlearn_regularize_all=True \
  +learner.iqlearn_log_every=500 \
  +learner.iqlearn_dynamics_coeff=1.0 \
  +learner.iqlearn_pretrain_deterministic_policy=True \
  +learner.iqlearn_pretrain_bc_coeff=1.0 \
  +learner.iqlearn_reward_clip=5.0 \
  +learner.iqlearn_reward_distill_steps="$IQLEARN_REWARD_DISTILL_STEPS" \
  +learner.iqlearn_reward_distill_batch_size=512 \
  +learner.demo_success_replay_path="$ROOT/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_bc_raw.pt" \
  +learner.demo_success_replay_fraction=0.50 \
  +learner.protect_demo_success_replay=True \
  +learner.demo_action_raw_limit="$action_limit" &
train_pid=$!

echo "[wait] waiting for Isaac training window"
xid=""
for _ in $(seq 1 240); do
  if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "[error] training exited before Isaac window appeared"
    tail -n 120 "$train_log" || true
    exit 1
  fi
  xid="$(find_isaac_xid || true)"
  [[ -n "$xid" ]] && break
  sleep 1
done

if [[ -z "$xid" ]]; then
  echo "[error] could not find Isaac window; refusing to record full desktop"
  xwininfo -root -tree | head -n 120 || true
  tail -n 120 "$train_log" || true
  exit 1
fi

sleep 2
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

train_status=0
wait "$train_pid" || train_status=$?
train_pid=""

kill -INT "$rec_pid" 2>/dev/null || true
wait "$rec_pid" 2>/dev/null || true
rec_pid=""

if [[ "$train_status" -ne 0 ]]; then
  echo "[error] training failed with status $train_status"
  tail -n 160 "$train_log" || true
  tail -n 120 "$ffmpeg_log" || true
  exit "$train_status"
fi

if grep -q "\[ERROR\] Exception occurred in training script" "$train_log"; then
  echo "[error] training log contains an exception; refusing to upload startup/failure video"
  tail -n 160 "$train_log" || true
  tail -n 120 "$ffmpeg_log" || true
  exit 1
fi

if [[ ! -s "$out" ]]; then
  echo "[error] no video output: $out"
  tail -n 120 "$ffmpeg_log" || true
  exit 1
fi

"$FFMPEG" -hide_banner -v error -i "$out" -f null - >/dev/null
echo "[ok] wrote live training video $(du -h "$out" | awk '{print $1}') $out"

echo "[upload] append live training video to W&B run id=$run_id"
"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python scripts/upload_video_to_existing_wandb_run.py \
    --run-id "$run_id" \
    --project "$WANDB_PROJECT" \
    --entity "$WANDB_ENTITY" \
    --video "$out" \
    --key "Training_Process/${run_id}_live_training" \
    --step "$NUM_ITERS" \
    --fps "$FPS"

echo "[done] live training video recorded and uploaded: $out"
