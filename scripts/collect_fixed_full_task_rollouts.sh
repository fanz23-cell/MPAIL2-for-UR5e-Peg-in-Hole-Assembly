#!/usr/bin/env bash
# 收集固定起始位置的完整任务 expert rollouts
# 用于训练只学固定任务的 MPAIL 实验
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

POLICY_PATH="${POLICY_PATH:-/path/to/your/full_task_teacher_policy.pt}"
OUTPUT_DIR="mpail2/train/isaac_franka/demos/fixed_full_task"
ROLLOUT_FILE="${OUTPUT_DIR}/fixed_full_rollouts_raw.pt"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] 收集固定起始位置的完整任务 rollouts..."
echo "[INFO] Policy: $POLICY_PATH"
echo "[INFO] Reset type: ObjectAnywhereEEAnywhere (完整任务，物体在桌上)"
echo "[INFO] 注意：这是固定起始位置，所有 episode 的物体和机器人初始位置一致"

~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
  python scripts/collect_omnireset_success_rollouts.py \
    --policy "$POLICY_PATH" \
    --output "$ROLLOUT_FILE" \
    --num-envs 16 \
    --target-success-episodes 50 \
    --max-env-steps 100000 \
    --seed 42 \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    --reset-type ObjectAnywhereEEAnywhere \
    --headless

echo ""
echo "[INFO] Rollouts 收集完成: $ROLLOUT_FILE"
echo "[INFO] 下一步：运行 build_fixed_full_task_demos.sh 来生成 MPAIL 训练用的 demo"
