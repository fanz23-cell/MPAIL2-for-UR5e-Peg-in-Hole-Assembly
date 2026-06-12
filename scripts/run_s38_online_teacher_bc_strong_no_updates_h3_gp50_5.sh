#!/usr/bin/env bash
# S37: teacher controls env; train policy only from online teacher obs/action pairs.
set -euo pipefail

cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

base=scripts/run_s36_teacher_policy_no_updates_diag_h3_gp50_5.sh
tmp=/tmp/run_s37_online_teacher_bc_no_updates.sh
sed 's/S36_teacherPolicyNoUpdatesDiag/S38_onlineTeacherBCStrongNoUpdates/g' "$base" > "$tmp"
sed -i 's/+learner.skip_updates=true \\/+learner.skip_updates=true \\\n  +learner.online_teacher_bc_steps=5000 \\\n  +learner.online_teacher_bc_batch_size=512 \\/' "$tmp"
bash "$tmp"
