#!/usr/bin/env bash
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

ckpt=${1:-}
penalty=${2:-10.0}
if [[ -z "$ckpt" ]]; then
  latest=$(find "$PWD/logs" -maxdepth 1 -type d -name 'obs22_S161clean60_S122base_mpailWarmstart_32env_30iter_*' | sort | tail -n 1)
  ckpt="$latest/models/model_30.pt"
fi
base_ckpt=${3:-"$PWD/logs/obs22_S122train_S115_rawObsTeacherBC_DAgger32_continue100_20260531_131911/models/model_100.pt"}
action_limit='[40,40,80,30,80,15,35]'
safe_penalty=${penalty//./p}
run_id="obs22_S167eval_S161_trueTrustRegion_penalty${safe_penalty}_32env_20iter_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"
echo "$ckpt" | tee "$run_dir/CKPT.txt"
echo "$base_ckpt" | tee "$run_dir/BASE_POLICY_CKPT.txt"
echo "$penalty" | tee "$run_dir/BASE_PLAN_DELTA_PENALTY.txt"

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
  +learner.demo_action_bc_coeff=5.0 \
  +learner.demo_action_bc_pretrain_steps=0 \
  +learner.demo_action_bc_direct_steps=0 \
  +learner.skip_updates=true \
  +learner.online_teacher_bc_steps=0 \
  +learner.demo_action_raw_limit="$action_limit" \
  +learner.planner_cfg.dynamics_cfg.residual=True \
  +learner.planner_cfg.dynamics_cfg.model_kwargs.override_last_layer_norm=True \
  +learner.planner_cfg.value_return_type=avg \
  +learner.planner_cfg.terminal_value_scale=0.0 \
  +learner.planner_cfg.reward_center_for_planning=True \
  +learner.planner_cfg.reward_clip_for_planning=5.0 \
  +learner.planner_cfg.base_plan_delta_penalty="$penalty" \
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
  +learner.mppi_policy_blend=0.05 \
  +learner.mppi_policy_delta_l2_max=0.03 \
  +learner.mppi_policy_delta_max=0.05 \
  +learner.mpail_policy_loss_coeff=0.0

jq -s 'def avg(k): ([.[] | .[k] // empty] | if length>0 then add/length else null end);
{
  n:length,
  success_mean:avg("Env/success_rate"),
  success_std:([.[] | ."Env/success_rate" // empty] as $xs | if ($xs|length)>0 then (($xs|add/length) as $m | (($xs|map((.-$m)*(.-$m))|add/length)|sqrt)) else null end),
  abnormal_mean:avg("Env/abnormal_robot_count"),
  reward_mppi_minus_policy:avg("PlannerDiag/reward_mppi_minus_policy"),
  reward_teacher_minus_mppi:avg("PlannerDiag/reward_teacher_minus_mppi"),
  q_mppi_minus_policy:((avg("PlannerDiag/q_mppi") // 0) - (avg("PlannerDiag/q_policy") // 0)),
  q_teacher_minus_mppi:avg("PlannerDiag/q_teacher_minus_mppi"),
  qstd_mppi:avg("PlannerDiag/qstd_mppi"),
  pred_progress_mppi_minus_policy:avg("PlannerDiag/pred_progress_mppi_minus_policy"),
  mppi_teacher_cos:avg("PlannerDiag/mppi_teacher_cos"),
  policy_teacher_cos:avg("PlannerDiag/policy_teacher_cos"),
  mppi_policy_cos:avg("PlannerDiag/mppi_policy_cos"),
  applied_delta:avg("MPPI/applied_delta_l2"),
  raw_delta:avg("MPPI/policy_delta_l2"),
  mean_base_delta_penalty:avg("MPPI/Mean Base Delta Penalty"),
  max_base_delta_penalty:avg("MPPI/Max Base Delta Penalty"),
  max_return:avg("MPPI/Max Return"),
  max_score_return:avg("MPPI/Max Score Return")
}' "$run_dir/metrics.jsonl" | tee "$run_dir/S167_EVAL_SUMMARY.json"
