#!/usr/bin/env bash
set -euo pipefail
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

# Parallel full-task S11 B/C/D diagnostic runs.
# raw_action_limit is intentionally unchanged: [8,8,8,5,5,1.25,0.5].

num_iterations=300
checkpoint_every=100
launcher_root="$PWD/logs/s11_bcd_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$launcher_root"

run_one() {
  local label="$1"
  local reward_every="$2"
  local num_elites="$3"
  local temperature="$4"

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
    +learner.use_terminations=False \
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

run_one "B_rewfreq1_elite128_temp2_termfalse" 1 128 2.0 > "$launcher_root/B.log" 2>&1 &
echo $! > "$launcher_root/B.pid"

run_one "C_rewfreq2_elite128_temp2_termfalse" 2 128 2.0 > "$launcher_root/C.log" 2>&1 &
echo $! > "$launcher_root/C.pid"

run_one "D_rewfreq2_elite128_temp3_termfalse" 2 128 3.0 > "$launcher_root/D.log" 2>&1 &
echo $! > "$launcher_root/D.pid"

echo "$launcher_root" | tee "$launcher_root/LAUNCHER_ROOT.txt"
wait
