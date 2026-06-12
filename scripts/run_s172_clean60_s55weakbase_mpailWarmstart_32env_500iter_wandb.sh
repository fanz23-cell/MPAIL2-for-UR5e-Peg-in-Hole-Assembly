#!/usr/bin/env bash
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

base_ckpt=${1:-"$PWD/logs/obs22_S55_MPAILModelWarmstart_skipPolicy_beta025_8env_30iter_20260529_011141/models/model_30.pt"}
action_limit='[40,40,80,30,80,15,35]'
run_id="obs22_S172clean60_S55weakbase_mpailWarmstart_32env_500iter_wandb_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"
echo "$base_ckpt" | tee "$run_dir/BASE_POLICY_CKPT.txt"

WANDB_PROJECT="${WANDB_PROJECT:-mpail2-omnireset}" \
WANDB_ENTITY="${WANDB_ENTITY:-fanz23-university-of-washington}" \
WANDB_MODE=online \
WANDB_NAME="$run_id" \
WANDB_DIR="$run_dir" \
NUM_ENVS=32 \
DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt \
LOG_FILE=$run_dir/train.log \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations=500 \
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
  omnireset_compact_obs_dim=60 \
  omnireset_raw_action_limit="$action_limit" \
  +omnireset_disable_domain_randomization=True \
  omnireset_reset_types=[ObjectAnywhereEEAnywhere] \
  omnireset_reset_probs=[1.0] \
  +learner.identity_encoder=True \
  +learner.replay_size=50000 \
  +learner.replay_ratio=0.078125 \
  +learner.replay_batch_size=256 \
  +learner.reward_learner_cfg.opt_params.lr=1e-6 \
  +learner.reward_learner_cfg.gp_coeff=50.0 \
  +learner.reward_learner_cfg.r1_coeff=0.05 \
  +learner.reward_update_every=1 \
  +learner.reward_center_for_planning=True \
  +learner.reward_center_mode=midpoint \
  +learner.reward_clip_for_planning=5.0 \
  +learner.obs_normalizer_cfg.std_floor=0.03 \
  +learner.obs_normalizer_cfg.skip_low_std_dims=False \
  +learner.raw_base_policy_checkpoint="$base_ckpt" \
  +learner.raw_base_policy_obs_key=raw_policy \
  +learner.raw_base_policy_obs_dim=215 \
  +learner.raw_base_policy_normalizer_demo_path="$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt" \
  +learner.raw_base_policy_normalizer_std_floor=0.03 \
  +learner.demo_action_bc_path="$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt" \
  +learner.demo_action_bc_coeff=5.0 \
  +learner.demo_action_bc_pretrain_steps=5000 \
  +learner.demo_action_bc_pretrain_batch_size=512 \
  +learner.demo_action_bc_direct_steps=0 \
  +learner.skip_updates=false \
  +learner.skip_reward_updates=false \
  +learner.skip_policy_updates=false \
  +learner.online_teacher_bc_steps=0 \
  +learner.demo_action_dyn_coeff=5.0 \
  +learner.demo_action_dyn_pretrain_steps=3000 \
  +learner.demo_action_dyn_pretrain_batch_size=512 \
  +learner.demo_action_multistep_dyn_coeff=10.0 \
  +learner.demo_action_multistep_dyn_horizon=3 \
  +learner.demo_action_rank_coeff=0.0 \
  +learner.demo_action_reward_rank_coeff=0.0 \
  +learner.online_mppi_reward_rank_coeff=0.0 \
  +learner.online_mppi_outcome_rank_coeff=0.0 \
  +learner.online_mppi_candidate_rank_coeff=0.0 \
  +learner.dyn_agent_coeff=0.25 \
  +learner.demo_action_raw_limit="$action_limit" \
  +learner.planner_cfg.dynamics_cfg.residual=True \
  +learner.planner_cfg.dynamics_cfg.model_kwargs.override_last_layer_norm=True \
  +learner.planner_cfg.value_return_type=avg \
  +learner.planner_cfg.terminal_value_scale=0.0 \
  +learner.planner_cfg.reward_center_for_planning=True \
  +learner.planner_cfg.reward_clip_for_planning=5.0 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.sampling_cfg.policy_proportion=0.0 \
  +learner.planner_cfg.sampling_cfg.max_std=0.025 \
  +learner.planner_cfg.sampling_cfg.min_std=0.0025 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=3 \
  +learner.loss_horizon=3 \
  +learner.mppi_residual_base_policy=true \
  +learner.mppi_action_library_anchor_policy_size=0 \
  +learner.mppi_action_library_size=0 \
  +learner.mppi_best_candidate_action=false \
  +learner.mppi_policy_blend=0.0 \
  +learner.mppi_policy_delta_l2_max=0.03 \
  +learner.mppi_policy_delta_max=0.05 \
  +learner.mpail_policy_loss_coeff=0.05 \
  +learner.policy_update_start_iter=5 \
  +learner.policy_trust_region_coeff=10.0

jq -s 'def avg(k): ([.[] | .[k] // empty] | if length > 0 then add / length else null end);
{
  n:length,
  abnormal_mean:avg("Env/abnormal_robot_count"),
  min_xyz_mean:avg("Env/min_xyz_mean"),
  ep_return_mean:avg("Env/ep_mean_return"),
  policy_teacher_cos:avg("TeacherDiag/policy_teacher_cos"),
  mppi_teacher_cos:avg("PlannerDiag/mppi_teacher_cos"),
  reward_mppi_minus_policy:avg("PlannerDiag/reward_mppi_minus_policy"),
  pred_progress_mppi_minus_policy:avg("PlannerDiag/pred_progress_mppi_minus_policy"),
  dyn_demo_loss:avg("DemoActionDyn/loss"),
  bc_loss:avg("Policy/demo_action_bc_loss"),
  mpail_policy_loss:avg("Policy/mpail_loss"),
  policy_alpha:avg("Policy/alpha"),
  value_loss:avg("Value/mean_loss")
}' "$run_dir/metrics.jsonl" | tee "$run_dir/S172_TRAIN_NOSUCCESS_SUMMARY.json"
