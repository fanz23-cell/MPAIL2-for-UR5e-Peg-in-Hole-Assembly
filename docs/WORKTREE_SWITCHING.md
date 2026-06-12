# Local Worktrees: Experiments vs Clean PR

This machine currently has two separate local directories for the OmniReset work.
They are intentionally separated so the experimental training code does not get
mixed into the clean PR for Tyler.

## 1. Experimental Workspace

Path:

```bash
/home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-
```

Use this directory for:

- Running OmniReset / MPAIL training experiments.
- Editing learner, runner, storage, reset handling, wrappers, scripts, and logs.
- Checking 50/100/500 iteration checkpoints.
- Watching policies in IsaacLab.
- Keeping messy local debugging state.

Go there with:

```bash
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-
git status --short --branch
```

This is the place to continue the 500-iteration experiments and inspect logs such
as:

```bash
logs/obs22_reset_semantics_R06a_500/
logs/obs22_R06a_forceclose_neg1_500/
```

## 2. Clean PR Workspace

Path:

```bash
/home/fanz23@netid.washington.edu/Desktop/mpail-research-omnireset-pr
```

Use this directory only for Tyler's clean PR branch:

```bash
fan/omnireset-peg-env-demos
```

This PR branch is based on `UWRobotLearning/mpail-research/main` and should stay
small and clean. It should only contain:

- OmniReset peg environment integration.
- OmniReset wrapper/config hooks needed to expose the environment.
- Official Route A demonstration files.
- Documentation for the demo files and how to run the entry point.

Do **not** use this directory for random training experiments. Do not copy
experimental learner/runner/storage/reset-semantics changes into this branch
unless Tyler explicitly asks for them.

Go there with:

```bash
cd /home/fanz23@netid.washington.edu/Desktop/mpail-research-omnireset-pr
git status --short --branch
```

Push PR updates from this directory:

```bash
git push --no-verify https://github.com/UWRobotLearning/mpail-research.git fan/omnireset-peg-env-demos
```

If Git asks for credentials:

```text
Username: fanz23-cell
Password: GitHub personal access token
```

## How To Switch

Switch by changing directories, not by `git checkout`:

```bash
# Back to experiments
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

# Back to clean PR
cd /home/fanz23@netid.washington.edu/Desktop/mpail-research-omnireset-pr
```

Always check where you are before running commands:

```bash
pwd
git status --short --branch
```

Quick rule:

```text
Mpail2-OmniReset-codex-       = experiment workspace
mpail-research-omnireset-pr   = clean PR workspace for Tyler
```
