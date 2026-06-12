#!/usr/bin/env bash
# 收集真正固定位置的完整任务 expert rollouts（使用 fixed_position override）
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

POLICY_PATH="mpail2/train/isaac_franka/demos/official_route_a/exported/policy.pt"
OUTPUT_DIR="mpail2/train/isaac_franka/demos/truly_fixed_full_task"
ROLLOUT_FILE="${OUTPUT_DIR}/truly_fixed_rollouts_raw.pt"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] =========================================="
echo "[INFO] 收集真正固定位置的完整任务 rollouts"
echo "[INFO] =========================================="
echo "[INFO] Policy: $POLICY_PATH"
echo "[INFO] Reset type: ObjectAnywhereEEAnywhere + FIXED POSITION OVERRIDE"
echo "[INFO] 所有 55 个 episode 的方块和插销位置将完全一致！"
echo "[INFO] =========================================="

# 使用修改后的收集脚本，添加 fixed_position 参数
~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
  python scripts/collect_omnireset_success_rollouts_fixed.py \
    --policy "$POLICY_PATH" \
    --output "$ROLLOUT_FILE" \
    --num-envs 16 \
    --target-success-episodes 55 \
    --max-env-steps 120000 \
    --seed 42 \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    --reset-type ObjectAnywhereEEAnywhere \
    --fixed-position \
    --headless

echo ""
echo "[INFO] =========================================="
echo "[INFO] Rollouts 收集完成: $ROLLOUT_FILE"
echo "[INFO] 下一步：运行 build_truly_fixed_demos.sh"
echo "[INFO] =========================================="
