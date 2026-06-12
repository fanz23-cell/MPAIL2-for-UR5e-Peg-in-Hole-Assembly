#!/usr/bin/env bash
# S35: teacher controls env; log policy mean vs MPPI final action on teacher states.
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

base=scripts/run_s34_teacher_diag_only_h3_gp50_5.sh
tmp=/tmp/run_s35_teacher_policy_vs_mppi_diag.sh
sed 's/S34_teacherDiagOnly/S35_teacherPolicyVsMppiDiag/g' "$base" > "$tmp"
bash "$tmp"
