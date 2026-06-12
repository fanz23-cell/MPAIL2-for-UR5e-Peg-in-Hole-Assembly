#!/usr/bin/env bash
# S13: 在 S12A (h=7, raw=6, std_floor=0.05) 基础上改
#   - std_floor: 0.05 -> 0.01  (放宽对低方差维度的钳制，让信息更接近原始分布)
#   - raw_action_limit: [6,6,6,4,4,1.0,0.5] -> [8,8,8,5,5,1.25,0.5]  (恢复 docs 里默认的 8/5/1.25)
# 其余配置严格沿用 S12A: 50 iter, 8env, gp=500, lr=1e-6, r1=0.05, replay 50k, ratio=0.078125, h=7, rollouts=512, elites=64
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

run_id="obs22_S13_fifo50k_stdfloor001_h7_raw8_gp500_lr1e6_r1c0p05_rewfreq1_8env_50iter_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"

WANDB_PROJECT=mpail2-omnireset \
WANDB_ENTITY=fanz23-university-of-washington \
WANDB_MODE=online \
WANDB_NAME="$run_id" \
NUM_ENVS=8 \
DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt \
LOG_FILE=$run_dir/train.log \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations=50 \
  log.run_name="$run_id" \
  log.run_log_dir="$run_dir" \
  log.checkpoint_every=50 \
  log.no_wandb=False \
  log.video=False \
  enable_cameras=False \
  omnireset_mpail_wrapper=True \
  omnireset_compact_obs_dim=22 \
  omnireset_force_gripper_close=False \
  +omnireset_disable_domain_randomization=True \
  omnireset_reset_types=[ObjectAnywhereEEAnywhere] \
  omnireset_reset_probs=[1.0] \
  omnireset_raw_action_limit=[8,8,8,5,5,1.25,0.5] \
  +learner.replay_size=50000 \
  +learner.replay_ratio=0.078125 \
  +learner.reward_learner_cfg.opt_params.lr=1e-6 \
  +learner.reward_learner_cfg.gp_coeff=500.0 \
  +learner.reward_learner_cfg.r1_coeff=0.05 \
  +learner.reward_update_every=1 \
  +learner.obs_normalizer_cfg.std_floor=0.01 \
  +learner.obs_normalizer_cfg.skip_low_std_dims=False \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
  +learner.loss_horizon=7
