#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
NUM_ENVS="${NUM_ENVS:-32}"
DEMO_PATH="${DEMO_PATH:-${REPO_ROOT}/mpail2/train/isaac_franka/demos/omnireset_peg_state_filtered.pt}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"

CONDA="${CONDA_EXE:-$(command -v conda 2>/dev/null || echo "${HOME}/miniconda3/bin/conda")}"
HEADLESS="${HEADLESS:-1}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-False}"
LOG_VIDEO="${LOG_VIDEO:-False}"
LIVE_RENDER="${LIVE_RENDER:-0}"
LIVE_RENDER_INTERVAL_STEPS="${LIVE_RENDER_INTERVAL_STEPS:-1}"
TRAIN_ENTRY="${TRAIN_ENTRY:-mpail2.train.train}"

if [[ "${ENABLE_CAMERAS}" == "1" || "${ENABLE_CAMERAS,,}" == "true" || "${ENABLE_CAMERAS,,}" == "yes" ]]; then
  export ENABLE_CAMERAS=1
  HYDRA_ENABLE_CAMERAS=True
else
  export ENABLE_CAMERAS=0
  HYDRA_ENABLE_CAMERAS=False
fi

CMD=(
  "${CONDA}" run --no-capture-output -n "${ENV_NAME}"
  python
)

if [[ "${TRAIN_ENTRY}" == *.py ]]; then
  CMD+=("${TRAIN_ENTRY}")
else
  CMD+=(-m "${TRAIN_ENTRY}")
fi

CMD+=(
  --env omnireset_peg
  log.no_wandb=True
  "log.video=${LOG_VIDEO}"
  "enable_cameras=${HYDRA_ENABLE_CAMERAS}"
  "num_envs=${NUM_ENVS}"
)

if [[ "${HEADLESS}" != "0" && "${HEADLESS,,}" != "false" && "${HEADLESS,,}" != "no" ]]; then
  CMD+=(--headless)
else
  CMD+=(headless=False)
fi

if [[ "${LIVE_RENDER}" == "1" || "${LIVE_RENDER,,}" == "true" || "${LIVE_RENDER,,}" == "yes" ]]; then
  CMD+=(runner.render_during_training=True)
  CMD+=("runner.render_interval_steps=${LIVE_RENDER_INTERVAL_STEPS}")
fi

if [[ -n "${DEMO_PATH}" ]]; then
  CMD+=("runner.path_to_demonstrations=${DEMO_PATH}")
fi

LOG_FILE="${LOG_FILE:-/tmp/mpail2_train.log}"
echo "[run_lab] logging to ${LOG_FILE}"
"${CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
