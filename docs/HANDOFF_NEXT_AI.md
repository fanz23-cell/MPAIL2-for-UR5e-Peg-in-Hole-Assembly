# OmniReset + MPAIL2 Current Handoff

Last updated: 2026-05-24

This file is the current authoritative handoff. Older experiments before the reset-semantics fix, gripper-free fix, and clean-eval fix should be treated as historical only. Do not copy old commands blindly.

## Current Goal

The user has been training MPAIL2 on OmniReset peg-in-hole. Until now most useful runs were the half-task / Route A setting: the object already starts grasped, and the robot learns the insertion segment.

The user now wants to switch to the full task: object starts on the table, robot must pick it up and insert it.

Important: switching to the full task is not just changing `DEMO_PATH`. You must also make the environment reset distribution, demo collection/filtering, wrapper observation, and evaluation task match the full-task setting.

## Repo / Environment

Working directory:

```bash
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-
```

Conda env:

```bash
~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset ...
```

Main scripts:

- Training launcher: `scripts/run_lab_omnireset.sh`
- Watch policy in IsaacLab GUI: `scripts/watch_mpail_omnireset.py`
- Existing handoff/status docs: `docs/HANDOFF_OMNIRESET_OBS22_STATUS_20260510.md`, `docs/WORKTREE_SWITCHING.md`

Current half-task demo:

```bash
mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt
```

Current play/eval task used for watching half-task:

```bash
OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0
```

## Fixes That Must Be Preserved

These are not optional. Do not regress them.

1. Reset semantics are fixed:
   - terminations and truncations are separated;
   - `terminal_observation` is used at reset boundaries;
   - abnormal transitions are not filtered out of replay, because MPAIL dynamics needs to learn that bad actions lead to bad states.

2. Gripper-free action is required:
   - do not force the gripper closed;
   - do not clamp gripper to a constant close command;
   - the policy/teacher/demo must provide the gripper action.

3. Clean evaluation is required, but use the fixed clean-eval behavior:
   - Tyler asked to use evaluation environment with domain randomization and sim2real randomization off;
   - an older implementation of `--disable-domain-randomization` broke the reset/gripper setup and caused severe jitter;
   - after the cleanfix, the policy no longer jitters. Preserve that fixed clean-eval logic.

4. MPAIL does not optimize the environment reward:
   - environment reward curves are diagnostics only;
   - MPAIL uses learned reward / discriminator reward from expert vs generated transitions;
   - do not tell Tyler that env reward shaping is the main training objective.

## Current Half-Task Baseline

The half-task means:

- reset starts with object already in the gripper;
- robot only needs to move the held object toward the hole and insert;
- demo path is `official_route_a/route_a_mpail.pt`;
- wrapper is low-dimensional obs22-style state;
- action is raw OSC action with gripper-free behavior.

The most recent parameter family:

- `gp_coeff=500`
- `replay_ratio=0.0391` for env8, or approximately `100 / (num_envs * 160)` when normalizing update count
- `num_rollouts=512`
- `num_elites=64`
- horizon `h=7`
- clean evaluation fixed
- gripper-free
- reset semantics fixed

Recent run naming examples:

```text
obs22_cleanfix_S1_gp500_8env_replay0p0391_1500iter_...
obs22_cleanfix_S1_gp500_32env_replay0p0391_100iter_...
```

## How To Switch From Half-Task To Full Task

Do all of these. Do not only change the demo path.

### 1. Collect Full-Task Expert Episodes

Use the official teacher / expert policy for the full peg-in-hole task, with object starting on the table.

The full-task demo must contain complete episodes:

- object starts on table;
- robot reaches;
- gripper closes and grasps;
- robot lifts/moves object;
- robot inserts into the hole;
- record `obs`, `action`, `next_obs`, `reward`, `terminated`, `truncated`, `done`, `success/info`.

Do not apply the old Route A insertion-only slicing. For full task, the pickup part must remain.

Only remove clearly useless post-success idle if it is appended after success. Do not remove failed-but-informative transitions unless they are corrupt / NaN / Inf.

Before training, show the user the new expert rollout quality in IsaacLab. The user wants to approve the demos before running long training.

### 2. Change The Reset Distribution / Task

Training and demo collection must use the same reset distribution.

For full task, do not use an `EEGrasped` reset type such as:

```text
ObjectAnywhereEEGrasped
ObjectRestingEEGrasped if it means already grasped
ObjectPartiallyAssembledEEGrasped
```

Use the reset where the object is on the table / resting ungrasped. The exact class/name must be verified from the UWLab task configs before launching.

The key rule:

```text
full-task demo reset == full-task training reset == full-task evaluation reset
```

If the env still starts with the object in the gripper, it is not the full task.

### 3. Recheck The Wrapper Observation

The current obs22 wrapper was tuned for half-task insertion. It may not contain enough information for grasping from the table.

For full task, the policy probably needs at least:

- end-effector pose/state;
- object/peg pose;
- hole pose;
- `ee_to_object`;
- `object_to_hole`;
- gripper opening or gripper joint state;
- possibly object-in-gripper/contact/grasp state if available;
- possibly EE velocity.

If the wrapper only tells the robot how to insert an already-held object, full-task training will fail even with good demos.

### 4. Keep Gripper-Free Action

For full task, gripper action is even more important.

Do not force close.
Do not force open.
Do not ignore the gripper dimension.

The action pipeline must allow:

- open before reaching;
- close for grasping;
- keep closed during transport;
- possibly release/relax near insertion if the teacher does so.

### 5. Use Clean Evaluation Without Breaking Reset

Use the cleanfix version of clean evaluation:

- domain randomization off;
- sim2real randomization off;
- reset semantics still correct;
- gripper initialization still correct;
- no old jitter-causing disable path.

If a policy suddenly jitters badly after enabling clean eval, suspect the clean-eval implementation, not immediately the training.

### 6. Train Short First

Do not immediately launch 2000 iterations on full task.

Recommended sequence:

1. collect full-task expert demos;
2. watch 5 to 10 expert episodes;
3. run a 20 to 50 iter smoke train;
4. watch `model_20.pt` or `model_50.pt`;
5. if action magnitude and exploration look sane, run 100 to 300 iter;
6. only then run 1000+ iter.

Full task is harder than half-task because the policy must learn grasping and insertion, not just insertion.

## How To Switch Back To Half-Task

To return to the previous half-task / Route A setup:

1. Set demo path back to:

```bash
DEMO_PATH=$PWD/mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt
```

2. Use the grasped/insertion reset distribution:

```text
ObjectAnywhereEEGrasped
```

or whichever current Route A reset class was used in the latest working half-task runs.

3. Use the half-task play/eval task:

```bash
--task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0
```

4. Keep:

- gripper-free;
- reset semantics fixed;
- cleanfix evaluation;
- obs22 low-dimensional wrapper unless deliberately testing another obs.

5. Use known half-task parameter family:

```text
gp_coeff=500
replay_ratio=0.0391 for env8
num_rollouts=512
num_elites=64
horizon=7
```

## Watching A Checkpoint

Use the watch script for interactive IsaacLab view. Example structure:

```bash
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
  python scripts/watch_mpail_omnireset.py \
  --checkpoint logs/<RUN>/<RUN_TIMESTAMP>/models/model_<ITER>.pt \
  --demo-path mpail2/train/isaac_franka/demos/official_route_a/route_a_mpail.pt \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
  --num-envs 1 \
  --num-episodes 5 \
  --disable-domain-randomization
```

Do not write broken flags like:

```text
--disable-domain-randomizationGrasped
```

That is one malformed argument.

For full task, replace the demo path and task/reset config with the full-task versions.

## W&B Sync

If W&B is in offline mode, sync with:

```bash
~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
  wandb sync logs/<RUN_GROUP>/<RUN_TIMESTAMP>/wandb/offline-run-*
```

The user wants run names to include the actual parameters:

```text
obs22_cleanfix_fulltask_gp500_8env_replay0p0391_100iter_<timestamp>
obs22_cleanfix_routeA_gp500_8env_replay0p0391_1000iter_<timestamp>
```

## What Tyler Has Asked / Diagnosed

Tyler’s important comments:

- make sure resets/terminations are handled correctly;
- separate truncations from terminations;
- do not filter abnormal transitions out of replay for MPAIL dynamics;
- use evaluation environment with domain randomization / sim2real things off;
- mean action magnitude was too low, suggesting not enough exploration / movement;
- videos of the training process and W&B logs are useful.

When explaining to Tyler, say:

```text
The current code keeps terminations and truncations separated, uses terminal_observation at reset boundaries, keeps abnormal transitions in replay, uses gripper-free actions, and uses the cleaned evaluation setup with domain randomization off without breaking reset initialization.
```

## Most Important Warning

For full task, changing only `DEMO_PATH` is not enough.

If full-task demos are used while the env still resets to already-grasped Route A, the training is not full task.

If full-task env is used while obs/action wrapper is still missing grasp-relevant state or clamps the gripper, the policy cannot learn pickup.

The correct switch is:

```text
full-task demo + full-task reset + full-task wrapper obs + gripper-free action + cleanfix eval + reset semantics fix
```
