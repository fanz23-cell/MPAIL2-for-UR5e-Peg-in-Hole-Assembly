#!/usr/bin/env bash
# 从固定起始位置的 rollouts 构建 MPAIL demo
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

INPUT_DIR="mpail2/train/isaac_franka/demos/fixed_full_task"
ROLLOUT_FILE="${INPUT_DIR}/fixed_full_rollouts_raw.pt"
RAW_OUTPUT="${INPUT_DIR}/fixed_full_bc_raw.pt"
MPAIL_OUTPUT="${INPUT_DIR}/fixed_full_mpail.pt"

if [[ ! -f "$ROLLOUT_FILE" ]]; then
  echo "[ERROR] Rollout file not found: $ROLLOUT_FILE"
  echo "[ERROR] 请先运行 collect_fixed_full_task_rollouts.sh"
  exit 1
fi

echo "[INFO] 从 rollouts 构建 MPAIL demo..."
echo "[INFO] Input: $ROLLOUT_FILE"
echo "[INFO] Raw output: $RAW_OUTPUT"
echo "[INFO] MPAIL output: $MPAIL_OUTPUT"

~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
  python scripts/build_omnireset_success_demo.py \
    --input "$ROLLOUT_FILE" \
    --raw-output "$RAW_OUTPUT" \
    --mpail-output "$MPAIL_OUTPUT" \
    --min-action-norm 0.05 \
    --min-pre-success-steps 5 \
    --include-success-transition

echo ""
echo "[INFO] Demo 构建完成！"
echo "[INFO] MPAIL 训练用 demo: $MPAIL_OUTPUT"
echo "[INFO] 下一步：运行 run_fixed_full_task_S1_100iter.sh 开始训练"
