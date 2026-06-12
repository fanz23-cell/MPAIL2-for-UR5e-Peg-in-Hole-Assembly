#!/usr/bin/env bash
# Run S25/S26 sequentially and evaluate model_50 checkpoints immediately.
set -uo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

mkdir -p logs/long_dynamics_queue
queue_log="logs/long_dynamics_queue/s25_s26_50_queue_$(date +%Y%m%d_%H%M%S).log"
echo "[queue] writing log to $queue_log" | tee "$queue_log"

run_and_eval() {
  local label="$1"
  local script="$2"
  local pattern="$3"
  local identity_args=("${@:4}")

  echo "[queue] START $label $(date)" | tee -a "$queue_log"
  if ! bash "$script" 2>&1 | tee -a "$queue_log"; then
    echo "[queue] FAILED $label $(date)" | tee -a "$queue_log"
    return 0
  fi

  local run_dir
  run_dir="$(ls -td logs/${pattern}_* 2>/dev/null | head -n 1)"
  if [[ -z "$run_dir" ]]; then
    echo "[queue] Could not locate run dir for $label" | tee -a "$queue_log"
    return 0
  fi

  local ckpt="$run_dir/models/model_50.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[queue] missing checkpoint $ckpt" | tee -a "$queue_log"
    return 0
  fi

  local out="logs/offline_checkpoint_eval_${label}_model50_teacher_actions.csv"
  echo "[queue] EVAL $label run_dir=$run_dir $(date)" | tee -a "$queue_log"
  ~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
    python scripts/offline_checkpoint_evaluator.py \
      --checkpoint "$ckpt" \
      --demo mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt \
      --device cpu \
      --sample-count 1024 \
      --reward-center-for-planning \
      --reward-clip-for-planning 5.0 \
      --std-floor 0.03 \
      "${identity_args[@]}" \
      --output "$out" 2>&1 | tee -a "$queue_log"

  echo "[queue] DONE $label $(date)" | tee -a "$queue_log"
}

run_and_eval "s25" "scripts/run_s25_identity_compact_h3_gp50_50.sh" "obs22_S25_identityCompact" \
  --identity-encoder --obs-dim 22 --latent-dim 22
run_and_eval "s26" "scripts/run_s26_action_rank_h3_gp50_50.sh" "obs22_S26_actionRankDyn"

echo "[queue] ALL DONE $(date)" | tee -a "$queue_log"
