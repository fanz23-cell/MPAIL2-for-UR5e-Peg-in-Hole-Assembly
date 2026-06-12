#!/usr/bin/env bash
# Run long dynamics-isolation experiments sequentially, and evaluate checkpoints
# after each run so the diagnostic table is ready when training finishes.
set -uo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

mkdir -p logs/long_dynamics_queue
queue_log="logs/long_dynamics_queue/s22_s24_queue_$(date +%Y%m%d_%H%M%S).log"
echo "[queue] writing log to $queue_log" | tee "$queue_log"

run_and_eval() {
  local label="$1"
  local script="$2"
  local pattern="$3"

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

  echo "[queue] EVAL $label run_dir=$run_dir $(date)" | tee -a "$queue_log"
  for step in 50 150 300; do
    local ckpt="$run_dir/models/model_${step}.pt"
    if [[ -f "$ckpt" ]]; then
      local out="logs/offline_checkpoint_eval_${label}_model${step}_teacher_actions.csv"
      ~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
        python scripts/offline_checkpoint_evaluator.py \
          --checkpoint "$ckpt" \
          --demo mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_bc_raw.pt \
          --device cpu \
          --sample-count 1024 \
          --reward-center-for-planning \
          --reward-clip-for-planning 5.0 \
          --std-floor 0.03 \
          --output "$out" 2>&1 | tee -a "$queue_log"
    else
      echo "[queue] missing checkpoint $ckpt" | tee -a "$queue_log"
    fi
  done
  echo "[queue] DONE $label $(date)" | tee -a "$queue_log"
}

run_and_eval "s22" "scripts/run_s22_demo_dyn_only_h3_gp50_300.sh" "obs22_S22_demoDynOnly"
run_and_eval "s23" "scripts/run_s23_demo_dyn10_h3_gp50_300.sh" "obs22_S23_demoDyn10"
run_and_eval "s24" "scripts/run_s24_residual_demo_dyn10_h3_gp50_300.sh" "obs22_S24_residualDemoDyn10"

echo "[queue] ALL DONE $(date)" | tee -a "$queue_log"
