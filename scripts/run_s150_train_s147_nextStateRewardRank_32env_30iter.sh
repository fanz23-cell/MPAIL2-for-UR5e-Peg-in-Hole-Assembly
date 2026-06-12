#!/usr/bin/env bash
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

ckpt=${1:-"$PWD/logs/obs22_S147train_S145_conservativeProgressDyn_32env_50iter_20260601_113551/models/model_50.pt"}
base_ckpt=${2:-"$PWD/logs/obs22_S122train_S115_rawObsTeacherBC_DAgger32_continue100_20260531_131911/models/model_100.pt"}
action_limit='[40,40,80,30,80,15,35]'
run_id="obs22_S150train_S147_nextStateRewardRank_32env_30iter_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"
echo "$ckpt" | tee "$run_dir/CKPT.txt"
echo "$base_ckpt" | tee "$run_dir/BASE_POLICY_CKPT.txt"

WANDB_MODE=disabled \
NUM_ENVS=32 \
DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt \
LOG_FILE=$run_dir/train.log \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations=30 \
  runner.resume_from_checkpoint="$ckpt" \
  runner.teacher_policy_path="$PWD/mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt" \
  runner.teacher_bootstrap_steps=0 \
  runner.teacher_blend_steps=0 \
  runner.teacher_diag_always=true \
  log.run_name="$run_id" \
  log.run_log_dir="$run_dir" \
  log.checkpoint_every=15 \
  log.no_wandb=True \
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
  +learner.demo_action_dyn_coeff=0.0 \
  +learner.demo_action_multistep_dyn_coeff=0.0 \
  +learner.demo_action_rank_coeff=0.0 \
  +learner.dyn_agent_coeff=0.0 \
  +learner.demo_action_raw_limit="$action_limit" \
  +learner.online_mppi_reward_rank_coeff=10.0 \
  +learner.online_mppi_reward_rank_margin=0.02 \
  +learner.online_mppi_reward_rank_mppi_weight=0.0 \
  +learner.online_mppi_reward_rank_policy_weight=0.0 \
  +learner.online_mppi_reward_rank_blended_weight=0.0 \
  +learner.online_mppi_next_state_rank_weight=25.0 \
  +learner.online_mppi_reward_rank_steps=8 \
  +learner.online_mppi_reward_rank_batch_size=512 \
  +learner.online_mppi_reward_rank_buffer_size=32768 \
  +learner.planner_cfg.reward_cfg.use_action=True \
  +learner.planner_cfg.dynamics_cfg.residual=True \
  +learner.planner_cfg.dynamics_cfg.model_kwargs.override_last_layer_norm=False \
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

jq -s '{
  n:length,
  success_mean:(map(."Env/success_rate")|add/length),
  reward_mppi_minus_policy:(map(."PlannerDiag/reward_mppi_minus_policy")|add/length),
  next_policy_minus_mppi:(map(."OnlineMPPIRank/next_policy_minus_mppi")|add/length),
  next_teacher_minus_mppi:(map(."OnlineMPPIRank/next_teacher_minus_mppi")|add/length),
  next_policy_gt_mppi:(map(."OnlineMPPIRank/next_policy_gt_mppi")|add/length),
  diag_action_only:(map(."PlannerDiag/reward_mppiAction_policyNext_minus_policy")|add/length),
  diag_next_only:(map(."PlannerDiag/reward_policyAction_mppiNext_minus_policy")|add/length)
}' "$run_dir/metrics.jsonl" | tee "$run_dir/S150_TRAIN_SUMMARY.json"
