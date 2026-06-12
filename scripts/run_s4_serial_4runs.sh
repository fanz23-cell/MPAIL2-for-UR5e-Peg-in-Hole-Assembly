#!/usr/bin/env bash
set -euo pipefail
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

run_one() {
  local label="$1"
  local gp="$2"
  local lr="$3"
  local run_id="obs22_${label}_replay0p078_rewfreq1_8env_50iter_$(date +%Y%m%d_%H%M%S)"
  local run_dir="$PWD/logs/$run_id"
  mkdir -p "$run_dir"
  echo "$run_id" | tee "$run_dir/RUN_ID.txt"

  WANDB_PROJECT=mpail2-omnireset \
  WANDB_ENTITY=fanz23-university-of-washington \
  WANDB_MODE=online \
  WANDB_NAME="$run_id" \
  NUM_ENVS=8 \
  DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt \
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
    omnireset_reset_types=[ObjectAnywhereEEGrasped] \
    omnireset_reset_probs=[1.0] \
    omnireset_raw_action_limit=[8,8,8,5,5,1.25,0.5] \
    +learner.replay_ratio=0.078125 \
    +learner.reward_learner_cfg.opt_params.lr="$lr" \
    +learner.reward_learner_cfg.gp_coeff="$gp" \
    +learner.reward_update_every=1 \
    +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
    +learner.planner_cfg.num_elites=64 \
    +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
    +learner.loss_horizon=7
}

# S4a: gp=10,  lr=1e-4
run_one "S4a_gp10_lr1e4"  10.0  1e-4

# S4b: gp=100, lr=1e-4
run_one "S4b_gp100_lr1e4" 100.0 1e-4

# S4c: gp=500, lr=3e-4
run_one "S4c_gp500_lr3e4" 500.0 3e-4

# S4d: gp=50,  lr=3e-4
run_one "S4d_gp50_lr3e4"  50.0  3e-4

echo "全部 4 组跑完"
