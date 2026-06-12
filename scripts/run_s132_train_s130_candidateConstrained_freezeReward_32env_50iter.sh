#!/usr/bin/env bash
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

ckpt=${1:-"$PWD/logs/obs22_S129train_S123_S122base_realOutcomeH5_narrowMPPI_32env_50iter_20260531_184807/models/model_50.pt"}
base_ckpt=${2:-"$PWD/logs/obs22_S122train_S115_rawObsTeacherBC_DAgger32_continue100_20260531_131911/models/model_100.pt"}
action_limit='[40,40,80,30,80,15,35]'
run_id="obs22_S132train_S130_candidateConstrained_freezeReward_32env_50iter_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"
echo "$ckpt" | tee "$run_dir/CKPT.txt"
echo "$base_ckpt" | tee "$run_dir/BASE_POLICY_CKPT.txt"

WANDB_PROJECT=mpail2-omnireset \
WANDB_ENTITY=fanz23-university-of-washington \
WANDB_MODE=online \
WANDB_NAME="$run_id" \
NUM_ENVS=32 \
DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt \
LOG_FILE=$run_dir/train.log \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations=50 \
  runner.resume_from_checkpoint="$ckpt" \
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
  omnireset_compact_obs_dim=119 \
  omnireset_raw_action_limit="$action_limit" \
  +omnireset_disable_domain_randomization=True \
  omnireset_reset_types=[ObjectAnywhereEEAnywhere] \
  omnireset_reset_probs=[1.0] \
  +learner.identity_encoder=True \
  +learner.replay_size=50000 \
  +learner.replay_ratio=0.078125 \
  +learner.obs_normalizer_cfg.std_floor=0.03 \
  +learner.obs_normalizer_cfg.skip_low_std_dims=False \
  +learner.raw_base_policy_checkpoint="$base_ckpt" \
  +learner.raw_base_policy_obs_key=raw_policy \
  +learner.raw_base_policy_obs_dim=215 \
  +learner.raw_base_policy_normalizer_demo_path="$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt" \
  +learner.raw_base_policy_normalizer_std_floor=0.03 \
  +learner.demo_action_bc_path="$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt" \
  +learner.demo_action_bc_coeff=0.0 \
  +learner.demo_action_bc_pretrain_steps=0 \
  +learner.demo_action_bc_direct_steps=0 \
  +learner.skip_policy_updates=true \
  +learner.skip_reward_updates=true \
  +learner.online_teacher_bc_steps=0 \
  +learner.demo_action_dyn_coeff=5.0 \
  +learner.demo_action_multistep_dyn_coeff=10.0 \
  +learner.demo_action_multistep_dyn_horizon=3 \
  +learner.demo_action_rank_coeff=5.0 \
  +learner.demo_action_rank_margin=0.02 \
  +learner.demo_action_reward_rank_coeff=0.0 \
  +learner.online_mppi_reward_rank_coeff=0.0 \
  +learner.online_mppi_outcome_rank_coeff=0.0 \
  +learner.dyn_agent_coeff=0.25 \
  +learner.demo_action_raw_limit="$action_limit" \
  +learner.planner_cfg.reward_cfg.use_action=True \
  +learner.planner_cfg.dynamics_cfg.residual=True \
  +learner.planner_cfg.value_return_type=avg \
  +learner.planner_cfg.terminal_value_scale=0.0 \
  +learner.planner_cfg.reward_center_for_planning=True \
  +learner.planner_cfg.reward_clip_for_planning=5.0 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.sampling_cfg.policy_proportion=0.50 \
  +learner.planner_cfg.sampling_cfg.max_std=0.025 \
  +learner.planner_cfg.sampling_cfg.min_std=0.0025 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=3 \
  +learner.loss_horizon=3 \
  +learner.mppi_residual_base_policy=true \
  +learner.mppi_action_library_anchor_policy_size=48 \
  +learner.mppi_action_library_size=16 \
  +learner.mppi_best_candidate_action=true \
  +learner.mppi_best_candidate_score_mode=first \
  +learner.mppi_best_candidate_policy_margin=0.0 \
  +learner.mppi_best_candidate_delta_penalty=0.5 \
  +learner.mppi_best_candidate_policy_cos_min=0.75 \
  +learner.mppi_policy_blend=0.05 \
  +learner.mppi_policy_delta_l2_max=0.03 \
  +learner.mppi_policy_delta_max=0.05 \
  +learner.mpail_policy_loss_coeff=0.0
