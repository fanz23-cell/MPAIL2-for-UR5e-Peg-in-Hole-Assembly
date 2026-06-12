#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_ID="${SWEEP_ID:-obs19_disable_abnormal_r06ab_500_$(date +%Y%m%d_%H%M%S)}"
NUM_ENVS="${NUM_ENVS:-16}"
ITERS="${ITERS:-500}"
DEMO_PATH="${DEMO_PATH:-${REPO_ROOT}/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt}"
ROOT_LOG_DIR="${ROOT_LOG_DIR:-${REPO_ROOT}/logs/obs19_disable_abnormal_r06ab_500/${SWEEP_ID}}"
SUMMARY_FILE="${ROOT_LOG_DIR}/summary.tsv"

export WANDB_PROJECT="${WANDB_PROJECT:-mpail2-omnireset}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${ROOT_LOG_DIR}"
{
  echo -e "run\tstatus\tlimits\tlog_dir"
} > "${SUMMARY_FILE}"

runs=(
  "R06a:6,6,6,4,4,1.0,0.5"
  "R06b:6,6,5,4,4,1.0,0.5"
)

echo "[sweep] id=${SWEEP_ID}"
echo "[sweep] root=${ROOT_LOG_DIR}"
echo "[sweep] demo=${DEMO_PATH}"
echo "[sweep] wandb_project=${WANDB_PROJECT} wandb_entity=${WANDB_ENTITY:-<default>} wandb_mode=${WANDB_MODE}"
echo "[sweep] num_envs=${NUM_ENVS} iters=${ITERS}"
echo "[sweep] compact_obs=19 includes ee_to_hole; abnormal_robot termination disabled; time_out kept"

for spec in "${runs[@]}"; do
  IFS=":" read -r run_name limits <<< "${spec}"
  run_dir="${ROOT_LOG_DIR}/${run_name}"
  log_file="${run_dir}/train.log"
  wandb_name="${SWEEP_ID}_${run_name}"
  mkdir -p "${run_dir}"

  echo "[sweep] START ${run_name} limits=[${limits}] $(date --iso-8601=seconds)"
  NUM_ENVS="${NUM_ENVS}" \
  DEMO_PATH="${DEMO_PATH}" \
  LOG_FILE="${log_file}" \
  bash scripts/run_lab_omnireset.sh \
    task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    runner.num_learning_iterations="${ITERS}" \
    log.run_name="${wandb_name}" \
    log.run_log_dir="${run_dir}" \
    log.checkpoint_every=50 \
    log.no_wandb=False \
    log.video=False \
    enable_cameras=False \
    omnireset_reset_types='[ObjectAnywhereEEGrasped]' \
    omnireset_reset_probs='[1.0]' \
    omnireset_disable_abnormal_termination=True \
    omnireset_mpail_wrapper=True \
    +learner.use_terminations=False \
    "omnireset_raw_action_limit=[${limits}]"

  status=$?
  echo -e "${run_name}\t${status}\t[${limits}]\t${run_dir}" >> "${SUMMARY_FILE}"
  echo "[sweep] END ${run_name} status=${status} $(date --iso-8601=seconds)"
done

echo "[sweep] DONE ${SWEEP_ID} $(date --iso-8601=seconds)"
echo "[sweep] summary=${SUMMARY_FILE}"
