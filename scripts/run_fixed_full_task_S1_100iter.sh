#!/usr/bin/env bash
# 固定起始位置完整任务训练 - S1 参数基线，100 iter
# 这是一个独立的实验配置，不影响现有训练文件
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

run_one() {
  local envs="$1"
  local iters="$2"
  local run_id="obs22_fixed_full_S1_gp500_lr1e4_replay0p039_${envs}env_${iters}iter_$(date +%Y%m%d_%H%M%S)"
  local run_dir="$PWD/logs/$run_id"
  mkdir -p "$run_dir"
  echo "$run_id" | tee "$run_dir/RUN_ID.txt"

  # 使用真正固定起始位置的完整任务 demo（55 个 episode，所有起始位置完全一致）
  local demo_path="$PWD/mpail2/train/isaac_franka/demos/truly_fixed_full_task/truly_fixed_mpail.pt"
  if [[ ! -f "$demo_path" ]]; then
    echo "[ERROR] Demo file not found: $demo_path"
    echo "[ERROR] 请先运行 build demo 脚本"
    exit 1
  fi

  echo "[INFO] =========================================="
  echo "[INFO] 固定起始位置完整任务训练"
  echo "[INFO] =========================================="
  echo "[INFO] Run ID: $run_id"
  echo "[INFO] Demo: $demo_path"
  echo "[INFO] Reset type: ObjectAnywhereEEAnywhere (完整任务，固定起始位置)"
  echo "[INFO] Task: 完整任务 (物体在桌上 -> 抓取 -> 插入)"
  echo "[INFO] Params: S1 baseline (gp=500, replay=0.0391, lr=1e-4, roll=512, elite=64, h=7)"
  echo "[INFO] Iterations: $iters"
  echo "[INFO] Envs: $envs"
  echo "[INFO] =========================================="

  WANDB_PROJECT=mpail2-omnireset \
  WANDB_ENTITY=fanz23-university-of-washington \
  WANDB_MODE=online \
  WANDB_NAME="$run_id" \
  NUM_ENVS="$envs" \
  DEMO_PATH="$demo_path" \
  LOG_FILE="$run_dir/train.log" \
  bash scripts/run_lab_omnireset.sh \
    task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    runner.num_learning_iterations="$iters" \
    runner.seed=42 \
    log.run_name="$run_id" \
    log.run_log_dir="$run_dir" \
    log.checkpoint_every=20 \
    log.no_wandb=False \
    log.video=False \
    enable_cameras=False \
    omnireset_mpail_wrapper=True \
    omnireset_compact_obs_dim=22 \
    omnireset_force_gripper_close=False \
    omnireset_fixed_position=True \
    +omnireset_disable_domain_randomization=True \
    omnireset_reset_types=[ObjectAnywhereEEAnywhere] \
    omnireset_reset_probs=[1.0] \
    omnireset_raw_action_limit=[8,8,8,5,5,1.25,0.5] \
    +learner.replay_ratio=0.0390625 \
    +learner.reward_learner_cfg.opt_params.lr=1e-4 \
    +learner.reward_learner_cfg.gp_coeff=500.0 \
    +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
    +learner.planner_cfg.num_elites=64 \
    +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
    +learner.loss_horizon=7
}

# 默认 8 env，100 iter
run_one "${1:-8}" "${2:-100}"
