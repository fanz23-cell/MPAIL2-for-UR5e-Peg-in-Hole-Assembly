#!/usr/bin/env bash
# Full task 训练 - S1 基线参数 + 大 raw_action_limit [35,32,63,25,59,12,26], 100 iter
# 基于 scripts/run_omnireset_obs22_gripperfree_explore5_500.sh 的 S1 配置改造为 full task
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

NUM_ENVS="${NUM_ENVS:-8}"
ITERS="${ITERS:-100}"

run_id="obs22_FullTask_S1_gp500_lr1e4_replay0p039_rawlimitBIG_${NUM_ENVS}env_${ITERS}iter_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"

DEMO_PATH="${DEMO_PATH:-$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt}"
if [[ ! -f "$DEMO_PATH" ]]; then
  echo "[ERROR] Demo file not found: $DEMO_PATH"
  exit 1
fi

echo "[INFO] =========================================="
echo "[INFO] Full Task 训练 (S1 基线参数 + 大 raw_action_limit)"
echo "[INFO] =========================================="
echo "[INFO] Run ID: $run_id"
echo "[INFO] Demo: $DEMO_PATH"
echo "[INFO] Reset type: ObjectAnywhereEEAnywhere (完整任务)"
echo "[INFO] Params: gp=500, replay=0.0391, reward_lr=1e-4, roll=512, elite=64, h=7"
echo "[INFO] raw_action_limit: [35, 32, 63, 25, 59, 12, 26]"
echo "[INFO] Iterations: $ITERS, Envs: $NUM_ENVS"
echo "[INFO] =========================================="

WANDB_PROJECT=mpail2-omnireset \
WANDB_ENTITY=fanz23-university-of-washington \
WANDB_MODE=online \
WANDB_NAME="$run_id" \
NUM_ENVS="$NUM_ENVS" \
DEMO_PATH="$DEMO_PATH" \
LOG_FILE="$run_dir/train.log" \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations="$ITERS" \
  log.run_name="$run_id" \
  log.run_log_dir="$run_dir" \
  log.checkpoint_every=20 \
  log.no_wandb=False \
  log.video=False \
  enable_cameras=False \
  omnireset_mpail_wrapper=True \
  omnireset_compact_obs_dim=22 \
  omnireset_force_gripper_close=False \
  omnireset_reset_types=[ObjectAnywhereEEAnywhere] \
  omnireset_reset_probs=[1.0] \
  omnireset_raw_action_limit=[35,32,63,25,59,12,26] \
  +learner.replay_ratio=0.0390625 \
  +learner.reward_learner_cfg.opt_params.lr=1e-4 \
  +learner.reward_learner_cfg.gp_coeff=500.0 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
  +learner.loss_horizon=7
