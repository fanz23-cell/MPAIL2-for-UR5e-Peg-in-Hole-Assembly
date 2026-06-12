#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-obs22_gripper_free_grasped_x2_2000_$(date +%Y%m%d_%H%M%S)}"
NUM_ENVS="${NUM_ENVS:-8}"
ITERS="${ITERS:-2000}"
DEMO_PATH="${DEMO_PATH:-${REPO_ROOT}/mpail2/train/isaac_franka/demos/official_route_a_x2/route_a_grasped_mpail.pt}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/logs/obs22_gripper_free_grasped_x2_2000/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"

export WANDB_PROJECT="${WANDB_PROJECT:-mpail2-omnireset}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${RUN_DIR}"

echo "[run] id=${RUN_ID}"
echo "[run] dir=${RUN_DIR}"
echo "[run] demo=${DEMO_PATH}"
echo "[run] task segment=ObjectAnywhereEEGrasped"
echo "[run] num_envs=${NUM_ENVS} iters=${ITERS}"
echo "[run] compact_obs=22 raw_limit=[6,6,6,4,4,1.0,0.5]"
echo "[run] force_gripper_close=False; gripper action is learned"
echo "[run] reset semantics fixed: terminated/truncated separated; terminal_observation used"

NUM_ENVS="${NUM_ENVS}" \
DEMO_PATH="${DEMO_PATH}" \
LOG_FILE="${LOG_FILE}" \
bash scripts/run_lab_omnireset.sh \
  task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  runner.num_learning_iterations="${ITERS}" \
  log.run_name="${RUN_ID}" \
  log.run_log_dir="${RUN_DIR}" \
  log.checkpoint_every=100 \
  log.no_wandb=False \
  log.video=False \
  enable_cameras=False \
  omnireset_reset_types='[ObjectAnywhereEEGrasped]' \
  omnireset_reset_probs='[1.0]' \
  omnireset_disable_abnormal_termination=False \
  omnireset_mpail_wrapper=True \
  omnireset_compact_obs_dim=22 \
  omnireset_force_gripper_close=False \
  +learner.use_terminations=True \
  "omnireset_raw_action_limit=[6,6,6,4,4,1.0,0.5]"
