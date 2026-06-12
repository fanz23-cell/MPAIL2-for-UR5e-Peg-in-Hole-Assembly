#!/usr/bin/env bash
set -euo pipefail
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

# S11 full-task direction sweep.
# Keeps raw_action_limit fixed at [8,8,8,5,5,1.25,0.5].
# Baseline reference: S10 h10, r1=0.05, reward_update_every=1,
# rollouts=512, elites=64, temperature=2.0.
num_iterations=300
checkpoint_every=100

run_one() {
  local label="$1"
  local reward_every="$2"
  local num_elites="$3"
  local temperature="$4"
  local use_terminations="$5"

  local run_id="obs22_S11_${label}_full_gp500_lr1e6_r1c0p05_rewfreq${reward_every}_replay0p078_8env_h10_${num_iterations}iter_$(date +%Y%m%d_%H%M%S)"
  local run_dir="$PWD/logs/$run_id"
  mkdir -p "$run_dir"
  echo "$run_id" | tee "$run_dir/RUN_ID.txt"

  WANDB_PROJECT=mpail2-omnireset \
  WANDB_ENTITY=fanz23-university-of-washington \
  WANDB_MODE=online \
  WANDB_NAME="$run_id" \
  NUM_ENVS=8 \
  DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_full_x2/full_object_anywhere_mpail.pt \
  LOG_FILE=$run_dir/train.log \
  bash scripts/run_lab_omnireset.sh \
    task=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    runner.num_learning_iterations="$num_iterations" \
    log.run_name="$run_id" \
    log.run_log_dir="$run_dir" \
    log.checkpoint_every="$checkpoint_every" \
    log.no_wandb=False \
    log.video=False \
    enable_cameras=False \
    omnireset_mpail_wrapper=True \
    omnireset_compact_obs_dim=22 \
    omnireset_force_gripper_close=False \
    +omnireset_disable_domain_randomization=True \
    'omnireset_reset_types=[ObjectAnywhereEEAnywhere]' \
    'omnireset_reset_probs=[1.0]' \
    'omnireset_raw_action_limit=[8,8,8,5,5,1.25,0.5]' \
    +learner.use_terminations="$use_terminations" \
    +learner.replay_ratio=0.078125 \
    +learner.reward_learner_cfg.opt_params.lr=1e-6 \
    +learner.reward_learner_cfg.gp_coeff=500.0 \
    +learner.reward_learner_cfg.r1_coeff=0.05 \
    +learner.reward_update_every="$reward_every" \
    +learner.planner_cfg.sampling_cfg.num_rollouts=512 \
    +learner.planner_cfg.num_elites="$num_elites" \
    +learner.planner_cfg.temperature="$temperature" \
    +learner.planner_cfg.sampling_cfg.num_timesteps=10 \
    +learner.loss_horizon=10
}

# A: Test whether S10's Value/MPPI drift came from reward targets changing too fast.
run_one "A_rewfreq2_elite64_temp2_termfalse" 2 64 2.0 False

# B: Isolate wider elites while keeping S10's reward update frequency.
run_one "B_rewfreq1_elite128_temp2_termfalse" 1 128 2.0 False

# C: Combine slower reward updates with wider elites.
run_one "C_rewfreq2_elite128_temp2_termfalse" 2 128 2.0 False

# D: If C still has high MPPI variance/abnormal behavior, make MPPI weighting more conservative.
run_one "D_rewfreq2_elite128_temp3_termfalse" 2 128 3.0 False

# E: Isolate conservative MPPI temperature under fast reward updates.
run_one "E_rewfreq1_elite128_temp3_termfalse" 1 128 3.0 False

# F: Test whether even wider elites reduce fake-high-score MPPI behavior.
run_one "F_rewfreq2_elite256_temp2_termfalse" 2 256 2.0 False

echo "S11 full-task 6-run direction sweep finished."
