#!/usr/bin/env bash
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

launcher_id="s174_seq_blend010_020_200iter_$(date +%Y%m%d_%H%M%S)"
launcher_dir="$PWD/logs/$launcher_id"
mkdir -p "$launcher_dir"
echo "$launcher_id" | tee "$launcher_dir/LAUNCHER_ID.txt"

run_one() {
  local blend="$1"
  local label="$2"
  echo "[S174] starting blend=${blend}" | tee -a "$launcher_dir/seq.log"
  bash scripts/run_s173_clean60_s55weakbase_mpailWarmstart_32env_200iter_wandb_blend.sh "$blend" \
    2>&1 | tee "$launcher_dir/${label}.log"
  local run_id
  run_id=$(grep -m1 '^obs22_S173clean60_' "$launcher_dir/${label}.log")
  echo "$run_id" | tee "$launcher_dir/${label}_RUN_ID.txt"
  echo "[S174] finished blend=${blend} run_id=${run_id}" | tee -a "$launcher_dir/seq.log"
}

run_one 0.10 blend010
run_one 0.20 blend020

jq -s '
  def avg(k): ([.[] | .[k] // empty] | if length > 0 then add / length else null end);
  def block(label; path):
    (path as $p | input_filename) as $ignored
    | {
        label: label,
        n:length,
        blend:avg("MPPI/blend"),
        min_xyz:avg("Env/min_xyz_mean"),
        abnormal:avg("Env/abnormal_robot_count"),
        ep_return:avg("Env/ep_mean_return"),
        avg_pos_align_error:avg("Metrics/task_command/average_pos_align_error"),
        end_pos_align_error:avg("Metrics/task_command/end_of_episode_pos_align_error"),
        dyn_loss:avg("Dyn/mean_loss"),
        dyn_raw_mae:avg("DynDiag/raw_mae"),
        reward_mppi_minus_policy:avg("PlannerDiag/reward_mppi_minus_policy"),
        pred_progress_mppi_minus_policy:avg("PlannerDiag/pred_progress_mppi_minus_policy"),
        q_mppi_minus_policy:(avg("PlannerDiag/q_mppi") - avg("PlannerDiag/q_policy")),
        applied_delta_l2:avg("MPPI/applied_delta_l2"),
        raw_delta_l2:avg("MPPI/policy_delta_l2"),
        mpail_policy_loss:avg("Policy/mpail_loss"),
        policy_alpha:avg("Policy/alpha"),
        value_loss:avg("Value/mean_loss")
      };
  block("placeholder"; "")
' /dev/null >/dev/null 2>&1 || true

python - <<'PY' "$launcher_dir"
import json
import sys
from pathlib import Path

launcher_dir = Path(sys.argv[1])
rows = []
for label in ("blend010", "blend020"):
    run_id = (launcher_dir / f"{label}_RUN_ID.txt").read_text().strip()
    metrics_path = Path("logs") / run_id / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    def avg(key):
        vals = [m[key] for m in metrics if key in m]
        return sum(vals) / len(vals) if vals else None
    rows.append({
        "label": label,
        "run_id": run_id,
        "n": len(metrics),
        "blend": avg("MPPI/blend"),
        "min_xyz": avg("Env/min_xyz_mean"),
        "abnormal": avg("Env/abnormal_robot_count"),
        "ep_return": avg("Env/ep_mean_return"),
        "avg_pos_align_error": avg("Metrics/task_command/average_pos_align_error"),
        "end_pos_align_error": avg("Metrics/task_command/end_of_episode_pos_align_error"),
        "dyn_loss": avg("Dyn/mean_loss"),
        "dyn_raw_mae": avg("DynDiag/raw_mae"),
        "reward_mppi_minus_policy": avg("PlannerDiag/reward_mppi_minus_policy"),
        "pred_progress_mppi_minus_policy": avg("PlannerDiag/pred_progress_mppi_minus_policy"),
        "q_mppi_minus_policy": (
            avg("PlannerDiag/q_mppi") - avg("PlannerDiag/q_policy")
            if avg("PlannerDiag/q_mppi") is not None and avg("PlannerDiag/q_policy") is not None
            else None
        ),
        "applied_delta_l2": avg("MPPI/applied_delta_l2"),
        "raw_delta_l2": avg("MPPI/policy_delta_l2"),
        "mpail_policy_loss": avg("Policy/mpail_loss"),
        "policy_alpha": avg("Policy/alpha"),
        "value_loss": avg("Value/mean_loss"),
    })

out = launcher_dir / "S174_BLEND010_020_COMPARISON_NOSUCCESS.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
PY
