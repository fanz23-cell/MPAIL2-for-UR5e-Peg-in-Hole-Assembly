#!/usr/bin/env bash
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

ckpt=${1:-}
if [[ -z "$ckpt" ]]; then
  latest=$(find "$PWD/logs" -maxdepth 1 -type d -name 'obs22_S154train_residualAdapter_S152v2_S122base_32env_20iter_*' | sort | tail -n 1)
  ckpt="$latest/models/model_20.pt"
fi
base_ckpt=${2:-"$PWD/logs/obs22_S122train_S115_rawObsTeacherBC_DAgger32_continue100_20260531_131911/models/model_100.pt"}
action_limit='[40,40,80,30,80,15,35]'
run_id="obs22_S155eval_residualAdapterCandidates_S154v2_S122base_32env_20iter_$(date +%Y%m%d_%H%M%S)"
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
  runner.num_learning_iterations=20 \
  runner.resume_from_checkpoint="$ckpt" \
  runner.teacher_policy_path="$PWD/mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt" \
  runner.teacher_bootstrap_steps=0 \
  runner.teacher_blend_steps=0 \
  runner.teacher_diag_always=true \
  log.run_name="$run_id" \
  log.run_log_dir="$run_dir" \
  log.checkpoint_every=20 \
  log.no_wandb=True \
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
  +learner.obs_normalizer_cfg.std_floor=0.03 \
  +learner.obs_normalizer_cfg.skip_low_std_dims=False \
  +learner.raw_base_policy_checkpoint="$base_ckpt" \
  +learner.raw_base_policy_obs_key=raw_policy \
  +learner.raw_base_policy_obs_dim=215 \
  +learner.raw_base_policy_normalizer_demo_path="$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt" \
  +learner.raw_base_policy_normalizer_std_floor=0.03 \
  +learner.demo_action_bc_path="$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt" \
  +learner.demo_action_bc_coeff=10.0 \
  +learner.demo_action_bc_pretrain_steps=0 \
  +learner.demo_action_bc_direct_steps=0 \
  +learner.skip_updates=true \
  +learner.online_teacher_bc_steps=0 \
  +learner.residual_adapter_enabled=true \
  +learner.residual_adapter_apply=false \
  +learner.residual_adapter_steps=0 \
  +learner.residual_adapter_batch_size=512 \
  +learner.residual_adapter_lr=1e-4 \
  +learner.residual_adapter_hidden_dim=256 \
  +learner.residual_adapter_scale=1.0 \
  +learner.mppi_residual_adapter_candidate_size=8 \
  +learner.mppi_residual_adapter_candidate_max_scale=1.0 \
  +learner.demo_action_raw_limit="$action_limit" \
  +learner.planner_cfg.dynamics_cfg.residual=True \
  +learner.planner_cfg.dynamics_cfg.model_kwargs.override_last_layer_norm=True \
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
  success_std:((map(."Env/success_rate") as $xs | ($xs|add/length) as $m | ($xs|map((.-$m)*(.-$m))|add/length|sqrt))),
  abnormal_mean:(map(."Env/abnormal_robot_count")|add/length),
  reward_mppi_minus_policy:(map(."PlannerDiag/reward_mppi_minus_policy")|add/length),
  mppi_teacher_cos:(map(."PlannerDiag/mppi_teacher_cos")|add/length),
  policy_teacher_cos:(map(."PlannerDiag/policy_teacher_cos")|add/length),
  applied_delta:(map(."MPPI/applied_delta_l2")|add/length),
  best_delta:(map(."MPPI/best_candidate_delta_l2")|add/length),
  anchor_rate:(map(."MPPI/best_candidate_anchor_rate")|add/length),
  best_minus_policy_anchor:(map(."MPPI/best_minus_policy_anchor")|add/length)
}' "$run_dir/metrics.jsonl" | tee "$run_dir/S155_EVAL_SUMMARY.json"
