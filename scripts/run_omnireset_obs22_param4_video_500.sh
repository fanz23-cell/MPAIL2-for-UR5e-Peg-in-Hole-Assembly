#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_ID="${SWEEP_ID:-obs22_param4_video_500_$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/logs/${SWEEP_ID}}"
NUM_ENVS="${NUM_ENVS:-8}"
ITERS="${ITERS:-500}"
DEMO_PATH="${DEMO_PATH:-${REPO_ROOT}/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt}"

export WANDB_PROJECT="${WANDB_PROJECT:-mpail2-omnireset}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

mkdir -p "${ROOT_DIR}"

run_one() {
  local tag="$1"
  shift

  local run_id="${SWEEP_ID}_${tag}"
  local run_dir="${ROOT_DIR}/${tag}"
  local log_file="${run_dir}/train.log"

  mkdir -p "${run_dir}"
  {
    echo "[sweep] id=${SWEEP_ID}"
    echo "[run] tag=${tag}"
    echo "[run] id=${run_id}"
    echo "[run] dir=${run_dir}"
    echo "[run] demo=${DEMO_PATH}"
    echo "[run] num_envs=${NUM_ENVS} iters=${ITERS}"
    echo "[run] training_video=False; checkpoint videos should be rendered with scripts/eval_omnireset.py"
    echo "[run] obs22 ObjectAnywhereEEGrasped force_close raw_limit=[8,8,8,5,5,1.25,0.5]"
    echo "[run] reset semantics: terminations/truncations separated; terminal_observation used; abnormal transitions not filtered"
    echo "[run] overrides=$*"
  } | tee "${run_dir}/run_info.txt"

  NUM_ENVS="${NUM_ENVS}" \
  DEMO_PATH="${DEMO_PATH}" \
  LOG_FILE="${log_file}" \
  bash scripts/run_lab_omnireset.sh \
    task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    runner.num_learning_iterations="${ITERS}" \
    log.run_name="${run_id}" \
    log.run_log_dir="${run_dir}" \
    log.checkpoint_every=50 \
    log.no_wandb=False \
    log.video=False \
    enable_cameras=False \
    omnireset_reset_types='[ObjectAnywhereEEGrasped]' \
    omnireset_reset_probs='[1.0]' \
    omnireset_disable_abnormal_termination=False \
    omnireset_mpail_wrapper=True \
    omnireset_compact_obs_dim=22 \
    omnireset_force_gripper_close=True \
    omnireset_gripper_close_value=-1.0 \
    +learner.use_terminations=True \
    "omnireset_raw_action_limit=[8,8,8,5,5,1.25,0.5]" \
    "$@"
}

{
  echo "[sweep] root=${ROOT_DIR}"
  echo "[sweep] wandb_project=${WANDB_PROJECT} wandb_mode=${WANDB_MODE}"
  echo "[sweep] Runs are sequential to avoid multiple IsaacLab renderers fighting for the same GPU."
  echo "[sweep] E1 replay_ratio; E3 horizon; E7 reward_lr; E9 combined."
} | tee "${ROOT_DIR}/sweep.log"

run_one "E1_replay_ratio" \
  +learner.replay_ratio=0.078125

run_one "E3_horizon10" \
  +learner.planner_cfg.sampling_cfg.num_timesteps=10

run_one "E7_reward_lr1e-4" \
  +learner.reward_learner_cfg.opt_params.lr=1e-4

run_one "E9_combo" \
  +learner.replay_ratio=0.078125 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=10 \
  +learner.planner_cfg.temperature=1.5 \
  +learner.reward_learner_cfg.opt_params.lr=1e-4

echo "[sweep] complete: ${ROOT_DIR}" | tee -a "${ROOT_DIR}/sweep.log"
