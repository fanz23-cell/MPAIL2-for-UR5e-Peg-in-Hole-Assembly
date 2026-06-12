#!/usr/bin/env bash
# S79b: matched pure-policy baseline for the S78 checkpoint.
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

ckpt=${1:-}
if [[ -z "$ckpt" ]]; then
  latest=$(find "$PWD/logs" -maxdepth 1 -type d -name 'obs22_S78_onPolicyMPPIRank_fromS76_policyExec_8env_20iter_*' | sort | tail -n 1)
  ckpt="$latest/models/model_20.pt"
fi
action_limit='[40,40,80,30,80,15,35]'
run_id="obs22_S79bEval_S78_purePolicy_8env_20iter_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"
echo "$ckpt" | tee "$run_dir/CKPT.txt"

WANDB_PROJECT=mpail2-omnireset \
WANDB_ENTITY=fanz23-university-of-washington \
WANDB_MODE=online \
WANDB_NAME="$run_id" \
NUM_ENVS=8 \
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
  log.no_wandb=False \
  log.video=False \
  enable_cameras=False \
  omnireset_mpail_wrapper=True \
  omnireset_compact_obs_dim=215 \
  omnireset_raw_action_limit="$action_limit" \
  +omnireset_disable_domain_randomization=True \
  omnireset_reset_types=[ObjectAnywhereEEAnywhere] \
  omnireset_reset_probs=[1.0] \
  +learner.identity_encoder=True \
  +learner.replay_size=50000 \
  +learner.replay_ratio=0.078125 \
  +learner.obs_normalizer_cfg.std_floor=0.03 \
  +learner.obs_normalizer_cfg.skip_low_std_dims=False \
  +learner.demo_action_bc_path="$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt" \
  +learner.demo_action_bc_coeff=5.0 \
  +learner.demo_action_bc_pretrain_steps=0 \
  +learner.demo_action_bc_direct_steps=999999 \
  +learner.skip_updates=true \
  +learner.online_teacher_bc_steps=0 \
  +learner.demo_action_dyn_coeff=0.0 \
  +learner.demo_action_dyn_pretrain_steps=0 \
  +learner.demo_action_rank_coeff=0.0 \
  +learner.demo_action_reward_rank_coeff=0.0 \
  +learner.dyn_agent_coeff=0.0 \
  +learner.demo_action_raw_limit="$action_limit" \
  +learner.planner_cfg.reward_cfg.use_action=True \
  +learner.planner_cfg.dynamics_cfg.residual=True \
  +learner.planner_cfg.value_return_type=avg \
  +learner.planner_cfg.terminal_value_scale=0.0 \
  +learner.planner_cfg.reward_center_for_planning=True \
  +learner.planner_cfg.reward_clip_for_planning=5.0 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.sampling_cfg.policy_proportion=0.95 \
  +learner.planner_cfg.sampling_cfg.max_std=0.08 \
  +learner.planner_cfg.sampling_cfg.min_std=0.01 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=3 \
  +learner.loss_horizon=3
