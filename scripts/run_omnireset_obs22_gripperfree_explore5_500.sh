#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_ID="${SWEEP_ID:-obs22_gripperfree_explore5_500_$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/logs/${SWEEP_ID}}"
NUM_ENVS="${NUM_ENVS:-8}"
ITERS="${ITERS:-500}"
DEMO_PATH="${DEMO_PATH:-${REPO_ROOT}/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt}"

export WANDB_PROJECT="${WANDB_PROJECT:-mpail2-omnireset}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

mkdir -p "${ROOT_DIR}"

run_one() {
  local tag="$1"
  local raw_limit="$2"
  shift 2

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
    echo "[run] obs22 ObjectAnywhereEEGrasped gripper_free raw_limit=${raw_limit}"
    echo "[run] reset semantics: terminations/truncations separated; terminal_observation used; abnormal transitions not filtered"
    echo "[run] no force close: omnireset_force_gripper_close=False"
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
    omnireset_force_gripper_close=False \
    +learner.use_terminations=True \
    "omnireset_raw_action_limit=${raw_limit}" \
    "$@"
}

{
  echo "[sweep] root=${ROOT_DIR}"
  echo "[sweep] wandb_project=${WANDB_PROJECT} wandb_mode=${WANDB_MODE}"
  echo "[sweep] sequential queue: one IsaacLab training process at a time for GPU stability"
  echo "[sweep] all runs are gripper-free; no forced close command"
} | tee "${ROOT_DIR}/sweep.log"

run_one "S1_balanced_replay0p039_rewardlr1e-4_roll512_elite64_h7" "[8,8,8,5,5,1.25,0.5]" \
  +learner.replay_ratio=0.0390625 \
  +learner.reward_learner_cfg.opt_params.lr=1e-4 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
  +learner.loss_horizon=7

run_one "S2_more_planner_roll1024_elite128_rewardlr1e-4_replay0p039_h7" "[8,8,8,5,5,1.25,0.5]" \
  +learner.replay_ratio=0.0390625 \
  +learner.reward_learner_cfg.opt_params.lr=1e-4 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=1024 \
  +learner.planner_cfg.num_elites=128 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
  +learner.loss_horizon=7

run_one "S3_slower_rewardlr5e-5_replay0p039_roll512_elite64_h7" "[8,8,8,5,5,1.25,0.5]" \
  +learner.replay_ratio=0.0390625 \
  +learner.reward_learner_cfg.opt_params.lr=5e-5 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
  +learner.loss_horizon=7

run_one "S4_big_action_limit10_rewardlr1e-4_replay0p039_roll512_elite64_h7" "[10,10,10,6,6,1.5,0.5]" \
  +learner.replay_ratio=0.0390625 \
  +learner.reward_learner_cfg.opt_params.lr=1e-4 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
  +learner.loss_horizon=7

run_one "S5_stable_value_replay0p0195_rewardlr1e-4_roll512_elite64_h7" "[8,8,8,5,5,1.25,0.5]" \
  +learner.replay_ratio=0.01953125 \
  +learner.reward_learner_cfg.opt_params.lr=1e-4 \
  +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
  +learner.planner_cfg.num_elites=64 \
  +learner.planner_cfg.sampling_cfg.num_timesteps=7 \
  +learner.loss_horizon=7

echo "[sweep] complete: ${ROOT_DIR}" | tee -a "${ROOT_DIR}/sweep.log"
