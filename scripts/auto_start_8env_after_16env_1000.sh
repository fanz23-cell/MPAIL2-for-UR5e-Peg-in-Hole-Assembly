#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-"
cd "$ROOT"

SOURCE_RUN="obs22_clean_eval_S1_gp500_16env_2000iter_20260521_213542"
SOURCE_DIR="$ROOT/logs/$SOURCE_RUN"
SOURCE_CKPT="$SOURCE_DIR/models/model_1000.pt"
WATCH_LOG="$ROOT/logs/${SOURCE_RUN}/auto_start_8env_after_1000.log"

log_msg() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$WATCH_LOG"
}

is_16env_alive() {
  pgrep -af "python -m mpail2.train.train.*${SOURCE_RUN}" >/dev/null 2>&1
}

kill_16env() {
  local pids
  pids="$(pgrep -f "python -m mpail2.train.train.*${SOURCE_RUN}" || true)"
  if [[ -n "$pids" ]]; then
    log_msg "Stopping 16env python pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 20
    pids="$(pgrep -f "python -m mpail2.train.train.*${SOURCE_RUN}" || true)"
    if [[ -n "$pids" ]]; then
      log_msg "Force-stopping remaining 16env python pids: $pids"
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

start_8env() {
  local run_id run_dir
  run_id="obs22_clean_eval_S1_gp500_8env_1000iter_$(date +%Y%m%d_%H%M%S)"
  run_dir="$ROOT/logs/$run_id"
  mkdir -p "$run_dir"

  log_msg "Starting 8env run: $run_id"
  (
    export NUM_ENVS=8
    export DEMO_PATH="$ROOT/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt"
    export LOG_FILE="$run_dir/train.log"
    export WANDB_MODE=online
    export WANDB_PROJECT=mpail2-omnireset
    export WANDB_NAME="$run_id"
    bash scripts/run_lab_omnireset.sh \
      task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
      runner.num_learning_iterations=1000 \
      log.run_name="$run_id" \
      log.run_log_dir="$run_dir" \
      log.checkpoint_every=50 \
      log.no_wandb=False \
      log.video=False \
      enable_cameras=False \
      omnireset_reset_types='[ObjectAnywhereEEGrasped]' \
      omnireset_reset_probs='[1.0]' \
      omnireset_disable_abnormal_termination=False \
      omnireset_disable_domain_randomization=True \
      omnireset_mpail_wrapper=True \
      omnireset_compact_obs_dim=22 \
      omnireset_force_gripper_close=False \
      +learner.use_terminations=True \
      omnireset_raw_action_limit='[8,8,8,5,5,1.25,0.5]' \
      +learner.replay_ratio=0.0390625 \
      +learner.reward_learner_cfg.opt_params.lr=1e-4 \
      +learner.reward_learner_cfg.gp_coeff=500.0 \
      +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
      +learner.planner_cfg.num_elites=64 \
      +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
      +learner.loss_horizon=7
  ) >>"$run_dir/launcher.out" 2>&1 &

  echo "$run_id" > "$ROOT/logs/latest_auto_8env_run.txt"
  log_msg "8env launcher PID=$!, run_dir=$run_dir"

  sleep 180
  if ! pgrep -af "python -m mpail2.train.train.*${run_id}" >/dev/null 2>&1; then
    log_msg "8env did not stay alive after launch. Check $run_dir/launcher.out"
    return 1
  fi
  log_msg "8env is alive."
}

log_msg "Watcher started. Waiting for $SOURCE_CKPT"
while [[ ! -f "$SOURCE_CKPT" ]]; do
  if ! is_16env_alive; then
    log_msg "16env process disappeared before model_1000.pt. Starting 8env fallback now."
    start_8env
    exit $?
  fi
  sleep 120
done

log_msg "Found model_1000.pt. Trying to run 8env concurrently with 16env."
start_8env

sleep 600
if ! is_16env_alive; then
  log_msg "16env is no longer alive after concurrent launch; cleaning remnants and keeping 8env."
  kill_16env
else
  log_msg "Both runs appear alive 10 minutes after concurrent launch. Leaving them running."
fi
