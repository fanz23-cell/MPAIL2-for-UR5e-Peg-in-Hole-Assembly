#!/usr/bin/env bash
# S36: teacher controls env; skip learner updates to isolate BC-pretrain action quality.
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

base=scripts/run_s35_teacher_policy_vs_mppi_diag_h3_gp50_5.sh
tmp=/tmp/run_s36_teacher_policy_no_updates_diag.sh
sed 's/S35_teacherPolicyVsMppiDiag/S36_teacherPolicyNoUpdatesDiag/g' "$base" > "$tmp"
sed -i 's/+runner.teacher_bootstrap_steps=1000000 \\/+runner.teacher_bootstrap_steps=1000000 \\\n  +learner.skip_updates=true \\/' "$tmp"
bash "$tmp"
