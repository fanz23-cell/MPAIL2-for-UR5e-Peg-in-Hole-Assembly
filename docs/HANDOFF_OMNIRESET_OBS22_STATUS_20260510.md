# OmniReset Peg-In-Hole MPAIL2 Handoff, 2026-05-10

This is the current state for the next AI/helper. The user is Fan. They are training OmniReset peg-in-hole in the MPAIL2 repo and need practical debugging, not high-level theory.

## Repo and Environment

- Repo: `/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-`
- Conda env: `mpail2-omnireset`
- Main training wrapper: `scripts/run_lab_omnireset.sh`
- Main watch/eval script: `scripts/watch_mpail_omnireset.py`
- Demo path currently used:
  - `mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt`
- Current task:
  - `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0`
- Current reset segment:
  - `ObjectAnywhereEEGrasped`

## Important Code Changes Already Made

1. `mpail2/envs/isaac/omnireset_wrapper.py`
   - Added compact observations.
   - Supported obs dims:
     - `13`: compact state with object/hole relation and core poses.
     - `19`: obs13 + `ee_to_hole`.
     - `22`: obs19 + `ee_vel_xyz`.
   - Added raw action scaling via `omnireset_raw_action_limit`.
   - Added `force_gripper_close` and `gripper_close_value`.
   - Correct close value is `-1.0`, not `+1.0`.

2. `mpail2/train/configs/isaac_base_config.py`
   - Added:
     - `omnireset_compact_obs_dim`
     - `omnireset_force_gripper_close`
     - `omnireset_gripper_close_value`
     - `omnireset_disable_abnormal_termination`

3. `mpail2/train/loops/isaac_image.py`
   - Applies OmniReset reset override.
   - Applies abnormal termination override.
   - Passes compact obs/action wrapper options into `OmniResetMPAILWrapper`.
   - Compacts demos to match compact obs dim.

4. `scripts/watch_mpail_omnireset.py`
   - Supports `--compact-obs-dim 22`.
   - Supports `--force-gripper-close --gripper-close-value -1.0`.

5. Added:
   - `scripts/debug_omnireset_gripper_sign.py`
   - `scripts/run_omnireset_obs22_disable_abnormal_r06a_500.sh`

## Critical Facts Learned

- The gripper sign was originally wrong.
  - IsaacLab `BinaryJointPositionAction`: negative action means close, nonnegative means open.
  - Correct force-close command is `-1.0`.
  - Previous `+1.0` force-close experiments actually forced the gripper open.

- The old `held~=` / `ee_peg < 0.08` heuristic is unreliable and should not be used as proof of whether the peg is held.
  - Official successful demos often have `ee_peg` around `0.22` to `0.72`.
  - Focus on visual behavior and `xyz` peg-hole distance instead.

- Using compact obs seems necessary.
  - Full OmniReset policy obs is 215-D, too large/noisy for early MPAIL dynamics/reward learning.
  - Franka/Kinova examples use much smaller state/action wrappers.

- Current best behavior before disabling abnormal termination:
  - `obs22 + R06a + force_close=-1.0 + abnormal_robot still active in env`
  - 500 iterations completed.
  - No successes.
  - In 10 watched episodes, about 3 showed real approach-to-hole behavior.
  - Best watched episode reached `xyz ~= 0.072`, but did not insert.
  - Main failure now is not immediate dropping; it is poor final alignment/insertion and inconsistent approach direction.

## Completed Run Before Current One

Run:

```text
logs/obs22_R06a_forceclose_neg1_500/obs22_R06a_forceclose_neg1_500_20260509_203639
```

Checkpoint:

```text
logs/obs22_R06a_forceclose_neg1_500/obs22_R06a_forceclose_neg1_500_20260509_203639/models/model_500.pt
```

Config:

- `compact_obs_dim=22`
- raw limit: `[6,6,6,4,4,1.0,0.5]`
- `force_gripper_close=True`
- `gripper_close_value=-1.0`
- `reset_types=[ObjectAnywhereEEGrasped]`
- `learner.use_terminations=False`
- Important: abnormal_robot was still active in the environment for this run.

Training summary:

- `Data/demo_transitions = 905`
- `Env/abnormal_robot_count = 0` at final summary
- `Env/done_count = 8`
- `Env/ep_mean_return = 2.1069`
- `Env/max_return = 2.29931`
- `Env/mean_length = 159.39`
- `Dyn/mean_loss = 0.03091`
- success remained `0`

W&B offline run:

```text
logs/obs22_R06a_forceclose_neg1_500/obs22_R06a_forceclose_neg1_500_20260509_203639/wandb/offline-run-20260509_203701-5cf3mkz4
```

Sync command:

```bash
wandb sync /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-/logs/obs22_R06a_forceclose_neg1_500/obs22_R06a_forceclose_neg1_500_20260509_203639/wandb/offline-run-20260509_203701-5cf3mkz4
```

Watched 10 episodes from this checkpoint:

1. `xyz 0.240 -> 0.457`, moved away, fail.
2. `0.142 -> 0.139`, almost no progress, fail.
3. `0.177 -> 0.393`, moved away, fail.
4. `0.258 -> 0.106`, clear approach, no insert.
5. `0.210 -> 0.153`, partial approach, no insert.
6. `0.140 -> 0.073`, best approach, no success.
7. `0.155 -> 0.243`, moved away, fail.
8. `0.171 -> 0.214`, oscillated, fail.
9. `0.295 -> 0.280`, little progress, fail.
10. `0.316 -> 0.207`, partial approach, no insert.

Conclusion: it learned some approach tendency, but not final insertion.

## Current Running Experiment

This is the important new run. It was started on 2026-05-10 around 16:52 PT.

Run:

```text
logs/obs22_disable_abnormal_R06a_500/obs22_disable_abnormal_R06a_500_20260510_165249
```

Script:

```text
scripts/run_omnireset_obs22_disable_abnormal_r06a_500.sh
```

Config:

- `ITERS=500`
- `NUM_ENVS=8`
- `compact_obs_dim=22`
- raw action limit: `[6,6,6,4,4,1.0,0.5]`
- `force_gripper_close=True`
- `gripper_close_value=-1.0`
- `reset_types=[ObjectAnywhereEEGrasped]`
- `omnireset_disable_abnormal_termination=True`
- `learner.use_terminations=False`
- W&B offline enabled

Sanity check confirmed before launching:

```text
[INFO] OmniReset termination override: disabled abnormal_robot; kept time_out enabled.
Termination Manager contains 1 active term:
  time_out True
```

Current run log:

```text
logs/obs22_disable_abnormal_R06a_500/obs22_disable_abnormal_R06a_500_20260510_165249/train.log
```

Nohup log:

```text
logs/obs22_disable_abnormal_R06a_500/obs22_disable_abnormal_R06a_500_20260510_165249/nohup.out
```

Check progress:

```bash
tail -f logs/obs22_disable_abnormal_R06a_500/obs22_disable_abnormal_R06a_500_20260510_165249/train.log
```

Check GPU:

```bash
nvidia-smi
```

Expected duration:

- Previous 500 iter with 8 envs took about 3.8 hours.
- This one should be similar, assuming no GPU contention.

After it finishes, watch the final checkpoint:

```bash
~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
  python scripts/watch_mpail_omnireset.py \
  --checkpoint logs/obs22_disable_abnormal_R06a_500/obs22_disable_abnormal_R06a_500_20260510_165249/models/model_500.pt \
  --demo-path mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  --num-envs 1 \
  --num-steps 1700 \
  --print-every 10 \
  --compact-wrapper \
  --compact-obs-dim 22 \
  --raw-action-limit 6 6 6 4 4 1.0 0.5 \
  --force-gripper-close \
  --gripper-close-value -1.0 \
  --reset-types ObjectAnywhereEEGrasped \
  --sleep 0.02
```

Watch about 10 episodes. Each episode is about 160 env steps. Use `xyz` as peg-hole distance. Success should be visible and `success=1.000` if achieved.

## What To Tell Tyler

Suggested message:

```text
Thanks, this helped. I checked the termination/reset issue more carefully.

For the latest completed run, I used 905 demo transitions from the official teacher rollout segments. The policy is not trained from the full task start; I reset into the ObjectAnywhereEEGrasped stage and train only the "object already grasped, move to hole and insert" segment.

Wrapper details:
- Original OmniReset policy obs is 215-D.
- I compact it to 22-D: object-in-hole relative pose, current/previous compact task pose features, ee_to_hole, and ee velocity xyz.
- I scale MPAIL's normalized action to conservative raw OSC limits [6,6,6,4,4,1.0,0.5].
- I force the gripper command closed. I verified the correct Robotiq close sign is -1.0; +1.0 actually opens the gripper.

The previous 500-iteration obs22 run had resets enabled and learner.use_terminations=False, but abnormal_robot was still active in the environment. It got some approach behavior but no successful insertions: in 10 watched episodes, the best one reduced peg-hole distance to about 0.07 m but did not insert.

I am now running a cleaner diagnostic with abnormal_robot termination fully disabled in the environment, keeping only time_out, still with learner.use_terminations=False. W&B is offline for now; I can sync/send the run directory.
```

Files/logs to send Tyler:

1. Completed previous W&B run:

```text
logs/obs22_R06a_forceclose_neg1_500/obs22_R06a_forceclose_neg1_500_20260509_203639/wandb/offline-run-20260509_203701-5cf3mkz4
```

2. Completed previous train log:

```text
logs/obs22_R06a_forceclose_neg1_500/obs22_R06a_forceclose_neg1_500_20260509_203639/train.log
```

3. Current cleaner run, after it finishes:

```text
logs/obs22_disable_abnormal_R06a_500/obs22_disable_abnormal_R06a_500_20260510_165249
```

4. The wrapper/code diff most relevant to Tyler:

```text
mpail2/envs/isaac/omnireset_wrapper.py
mpail2/train/loops/isaac_image.py
mpail2/train/configs/isaac_base_config.py
scripts/watch_mpail_omnireset.py
scripts/run_omnireset_obs22_disable_abnormal_r06a_500.sh
```

## Next Recommended Steps

1. Let the current 500-iter disable-abnormal run finish.
2. Inspect final W&B/train curves:
   - success
   - ep_mean_return
   - mean_length
   - done_count
   - time_out_count
   - abnormal_robot_count should stay 0 because term is removed
   - reward/discriminator separation
   - dyn loss
3. Watch 10 episodes from `model_500.pt`.
4. If still no success but it approaches:
   - Ask Tyler whether MPAIL should be run with a different transfer/scratch setup for OmniReset.
   - Consider adding final alignment features or a two-stage reset distribution closer to the hole.
   - Consider using more demonstrations: current training only has 905 transitions, which Tyler implied may be low.
5. If it still moves away frequently:
   - Check whether action dimensions/signs for OSC xyz/rpy are all correct, not just gripper.
   - Use `scripts/debug_omnireset_arm_sign.py` or create a more systematic action sign probe.

