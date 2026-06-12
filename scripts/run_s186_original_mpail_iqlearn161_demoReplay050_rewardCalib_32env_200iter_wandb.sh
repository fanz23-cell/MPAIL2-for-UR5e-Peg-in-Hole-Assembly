#!/usr/bin/env bash
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

action_limit='[40,40,80,30,80,15,35]'
run_id="obs215_S186_originalMPAIL_officialIQlearn161ep_demoReplay050_rewardCalib_32env_200iter_wandb_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"

WANDB_PROJECT=mpail2-omnireset \
WANDB_ENTITY=fanz23-university-of-washington \
WANDB_MODE=online \
WANDB_NAME="$run_id" \
NUM_ENVS=32 \
DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_mpail.pt \
LOG_FILE=$run_dir/train.log \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations=200 \
  runner.teacher_policy_path="$PWD/mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt" \
  runner.teacher_bootstrap_steps=0 \
  runner.teacher_blend_steps=0 \
  runner.teacher_diag_always=true \
  log.run_name="$run_id" \
  log.run_log_dir="$run_dir" \
  log.checkpoint_every=25 \
  log.no_wandb=False \
  log.video=False \
  enable_cameras=False \
  omnireset_mpail_wrapper=True \
  omnireset_compact_obs_dim=215 \
  omnireset_raw_action_limit="$action_limit" \
  +omnireset_disable_domain_randomization=True \
  omnireset_reset_types=[ObjectAnywhereEEAnywhere] \
  omnireset_reset_probs=[1.0] \
  +learner.iqlearn_pretrain_path="$PWD/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_bc_raw.pt" \
  +learner.iqlearn_pretrain_steps=10000 \
  +learner.iqlearn_pretrain_batch_size=512 \
  +learner.iqlearn_alpha=0.5 \
  +learner.iqlearn_init_temp=0.01 \
  +learner.iqlearn_learn_temp=False \
  +learner.iqlearn_policy_batch_ratio=1.0 \
  +learner.iqlearn_regularize_all=True \
  +learner.iqlearn_log_every=500 \
  +learner.iqlearn_dynamics_coeff=1.0 \
  +learner.iqlearn_reward_distill_steps=0 \
  +learner.demo_success_replay_path="$PWD/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_bc_raw.pt" \
  +learner.demo_success_replay_fraction=0.50 \
  +learner.protect_demo_success_replay=True \
  +learner.reward_center_for_planning=True \
  +learner.reward_center_mode=midpoint \
  +learner.reward_clip_for_planning=5.0 \
  +learner.demo_action_raw_limit="$action_limit"
