#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MPAIL3_ROOT="${MPAIL_SOURCE_ROOT:-$ROOT/snapshots/mpail-research}"

TASK_DEMO="${OMNIRESET_TASK_DEMO_PATH:-$ROOT/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_mpail.pt}"
BC_DEMO="${OMNIRESET_BC_DEMO_PATH:-$ROOT/mpail2/train/isaac_franka/demos/official_full_150ep/full_object_anywhere_bc_raw.pt}"
TEACHER_POLICY="${OMNIRESET_TEACHER_POLICY_PATH:-$ROOT/mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt}"

for required_file in "$TASK_DEMO" "$BC_DEMO" "$TEACHER_POLICY"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required local demo/policy file not found: $required_file" >&2
    echo "Set OMNIRESET_TASK_DEMO_PATH, OMNIRESET_BC_DEMO_PATH, and OMNIRESET_TEACHER_POLICY_PATH." >&2
    exit 2
  fi
done
if [[ ! -d "$MPAIL3_ROOT" ]]; then
  echo "MPAIL source tree not found: $MPAIL3_ROOT" >&2
  echo "Set MPAIL_SOURCE_ROOT to the later source snapshot." >&2
  exit 2
fi

action_limit='[40,40,80,30,80,15,35]'
run_id="obs215_S191_mpail3_tylerIQlearnPretrain_rewardTarget_fixedStdBC_demoReplay050_32env_100iter_wandb_$(date +%Y%m%d_%H%M%S)"
run_dir="$ROOT/logs/$run_id"
mkdir -p "$run_dir"
mkdir -p /tmp/mpl-cache
echo "$run_id" | tee "$run_dir/RUN_ID.txt"

cd "$MPAIL3_ROOT"

WANDB_PROJECT="${WANDB_PROJECT:-mpail2-omnireset}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
WANDB_NAME="$run_id" \
NUM_ENVS=32 \
MPLCONFIGDIR=/tmp/mpl-cache \
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${CONDA_EXE:-$HOME/miniconda3/bin/conda}" run --no-capture-output -n mpail2-omnireset \
python -m mpail2.train.train \
  --env omnireset_peg \
  --headless \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations=100 \
  runner.path_to_demonstrations="$TASK_DEMO" \
  runner.teacher_policy_path="$TEACHER_POLICY" \
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
  +learner.iqlearn_pretrain_path="$BC_DEMO" \
  +learner.iqlearn_pretrain_steps=10000 \
  +learner.iqlearn_pretrain_batch_size=512 \
  +learner.iqlearn_alpha=0.5 \
  +learner.iqlearn_init_temp=0.01 \
  +learner.iqlearn_learn_temp=False \
  +learner.iqlearn_policy_batch_ratio=1.0 \
  +learner.iqlearn_regularize_all=True \
  +learner.iqlearn_log_every=500 \
  +learner.iqlearn_dynamics_coeff=1.0 \
  +learner.iqlearn_pretrain_deterministic_policy=True \
  +learner.iqlearn_pretrain_bc_coeff=1.0 \
  +learner.iqlearn_reward_clip=5.0 \
  +learner.iqlearn_reward_distill_steps=2000 \
  +learner.iqlearn_reward_distill_batch_size=512 \
  +learner.demo_success_replay_path="$BC_DEMO" \
  +learner.demo_success_replay_fraction=0.50 \
  +learner.protect_demo_success_replay=True \
  +learner.demo_action_raw_limit="$action_limit" \
  2>&1 | tee "$run_dir/train.log"
