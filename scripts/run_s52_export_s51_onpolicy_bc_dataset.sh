#!/usr/bin/env bash
# S52: export beta=0.25 on-policy DAgger states/actions from the best S51 policy.
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

ckpt=${1:-"logs/obs22_S51_policyDAggerResidual025_fromS50_8env_25iter_20260529_000226/models/model_25.pt"}
bc_path=${2:-"logs/obs22_S49_exportResidualControllerBC_8env_15iter_20260528_230919/s49_residual_controller_bc_TRUE_raw.pt"}
action_limit='[40,40,80,30,80,15,35]'
run_id="obs22_S52_exportS51OnPolicyBC_beta025_8env_15iter_$(date +%Y%m%d_%H%M%S)"
run_dir="$PWD/logs/$run_id"
mkdir -p "$run_dir"
echo "$run_id" | tee "$run_dir/RUN_ID.txt"
echo "$ckpt" | tee "$run_dir/CKPT.txt"
echo "$bc_path" | tee "$run_dir/BC_PATH.txt"

WANDB_PROJECT=mpail2-omnireset \
WANDB_ENTITY=fanz23-university-of-washington \
WANDB_MODE=online \
WANDB_NAME="$run_id" \
NUM_ENVS=8 \
DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt \
LOG_FILE=$run_dir/train.log \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.resume_from_checkpoint="$ckpt" \
  runner.num_learning_iterations=15 \
  runner.teacher_policy_path="$PWD/mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt" \
  runner.teacher_bootstrap_steps=999999 \
  runner.teacher_blend_steps=0 \
  runner.policy_control_after_online_teacher_bc=true \
  runner.teacher_diag_always=true \
  runner.teacher_dagger_collect=true \
  runner.teacher_dagger_buffer_size=50000 \
  runner.teacher_dagger_export_path="s52_s51_onpolicy_bc_raw.pt" \
  runner.teacher_dagger_label_mode=executed \
  runner.teacher_residual_beta=0.25 \
  log.run_name="$run_id" \
  log.run_log_dir="$run_dir" \
  log.checkpoint_every=5 \
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
  +learner.demo_action_bc_path="$bc_path" \
  +learner.demo_action_bc_coeff=5.0 \
  +learner.demo_action_bc_pretrain_steps=0 \
  +learner.demo_action_bc_direct_steps=999999 \
  +learner.skip_updates=true \
  +learner.online_teacher_bc_steps=3000 \
  +learner.online_teacher_bc_batch_size=512 \
  +learner.demo_action_dyn_coeff=0.0 \
  +learner.demo_action_dyn_pretrain_steps=0 \
  +learner.demo_action_rank_coeff=0.0 \
  +learner.dyn_agent_coeff=0.0 \
  +learner.demo_action_raw_limit="$action_limit" \
  +learner.planner_cfg.sampling_cfg.num_timesteps=3 \
  +learner.loss_horizon=3
