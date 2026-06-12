#!/usr/bin/env bash
# S24: Same as S23, but dynamics predicts latent delta: z_next = z + f(z, a).
# Tests whether absolute next-latent prediction is hiding action-conditioned
# direction behind a large state reconstruction term.
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

run_id="obs22_S24_residualDemoDyn10_h3_policy50_fifo50k_stdfloor003_raw8_gp50_lr1e6_r1c0p05_clip5_rewfreq1_8env_300iter_$(date +%Y%m%d_%H%M%S)"
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
  runner.num_learning_iterations=300 \
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
  +learner.reward_learner_cfg.gp_coeff=50.0 \
  +learner.reward_learner_cfg.r1_coeff=0.05 \
  +learner.reward_update_every=1 \
  +learner.reward_center_for_planning=True \
  +learner.reward_center_mode=midpoint \
  +learner.reward_clip_for_planning=5.0 \
  +learner.obs_normalizer_cfg.std_floor=0.03 \
  +learner.obs_normalizer_cfg.skip_low_std_dims=False \
  +learner.demo_action_bc_path="$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt" \
  +learner.demo_action_bc_coeff=1.0 \
  +learner.demo_action_dyn_coeff=10.0 \
  +learner.dyn_agent_coeff=1.0 \
  +learner.demo_action_raw_limit=[8,8,8,5,5,1.25,0.5] \
  +learner.planner_cfg.dynamics_cfg.residual=True \
  +learner.planner_cfg.value_return_type=avg \
  +learner.planner_cfg.terminal_value_scale=0.0 \
  +learner.planner_cfg.reward_center_for_planning=True \
  +learner.planner_cfg.reward_clip_for_planning=5.0 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.sampling_cfg.policy_proportion=0.5 \
  +learner.planner_cfg.sampling_cfg.max_std=1.0 \
  +learner.planner_cfg.sampling_cfg.min_std=0.03 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=3 \
  +learner.loss_horizon=3
