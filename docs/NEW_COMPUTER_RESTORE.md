# New Computer Restore Handoff

Last updated: 2026-06-11

This file is for moving the current OmniReset/MPAIL2 experiment workspace to a new computer.

## What Is Already Backed Up

The external drive was formatted as ext4 and labeled:

```bash
MPAIL_BACKUP
```

The current workspace was copied to:

```bash
/media/fanz23/MPAIL_BACKUP/Mpail2-OmniReset-codex-backups/Mpail2-OmniReset-codex-
```

The backup includes the full repo directory, including:

- `.git/`
- tracked code
- untracked scripts
- local demos
- `logs/`
- checkpoints
- videos
- local `wandb/`
- environment export files

The final checked size was approximately:

```text
source: 55G
backup: 55G
```

The final sync completed with `to-chk=0`.

## Important Limit

The external drive preserves the experiment workspace, not a guaranteed runnable Linux system image.

On the new computer, rebuild the runtime environment. Do not expect a copied conda environment or IsaacSim native libraries to be portable across machines.

The new computer still needs:

- NVIDIA driver
- CUDA-compatible PyTorch
- IsaacSim / IsaacLab dependencies used by this repo
- the `mpail2-omnireset` conda environment
- working GUI/X11 or headless rendering setup, depending on the command

## Restore Workspace On New Computer

Plug in the external drive, then check where it mounted:

```bash
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS
```

If the drive is mounted as `/media/$USER/MPAIL_BACKUP`, restore with:

```bash
mkdir -p ~/Desktop

rsync -aH --info=progress2 \
/media/$USER/MPAIL_BACKUP/Mpail2-OmniReset-codex-backups/Mpail2-OmniReset-codex- \
~/Desktop/Mpail2-OmniReset-codex-
```

Then enter the repo:

```bash
cd ~/Desktop/Mpail2-OmniReset-codex-
```

## Verify The Restored Workspace

Run:

```bash
du -sh .
git status --short --branch
ls -lh envs/mpail2-omnireset-current-no-builds.yml envs/mpail2-omnireset-current-explicit.txt
test -d logs && echo logs-ok
test -d wandb && echo wandb-ok
test -d mpail2/train/isaac_franka/demos/official_full_150ep && echo demos-ok
```

Expected rough result:

```text
workspace size: about 55G
logs-ok
wandb-ok
demos-ok
```

`git status` is expected to be dirty because the current experiment state contains many uncommitted and untracked files.

## Rebuild Conda Environment

Preferred first attempt:

```bash
bash scripts/bootstrap_lab_omnireset.sh
```

If the bootstrap script is not enough, use the exported environment record:

```bash
~/miniconda3/bin/conda env create -f envs/mpail2-omnireset-current-no-builds.yml
```

The explicit export is also saved for reference:

```bash
envs/mpail2-omnireset-current-explicit.txt
```

That explicit file is best treated as a package-version reference, not the first restore path, because exact package URLs can be machine/platform sensitive.

## Minimal Runtime Checks

Check Python and CUDA:

```bash
~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:

```text
torch version prints
True
```

Then check that the repo imports:

```bash
~/miniconda3/bin/conda run --no-capture-output -n mpail2-omnireset \
python -c "import mpail2; print('mpail2-ok')"
```

## First Isaac/Viewer Check

Use a known playback command before launching new training. For example, inspect a known S190/S192-style checkpoint only after verifying the exact checkpoint path exists in `logs/`.

Find candidate checkpoints:

```bash
find logs -path '*models/model_*.pt' | rg 'S190|S192|S189|S185' | tail -n 20
```

Then run the matching watcher script used for that checkpoint family.

If the viewer fails, diagnose environment/Isaac paths before changing algorithm code.

## Continue The Existing Experiment Thread

To continue the same Codex/ChatGPT debugging thread:

1. Use the same ChatGPT/Codex account.
2. Open this same conversation if possible.
3. Restore the repo to:

```bash
~/Desktop/Mpail2-OmniReset-codex-
```

4. Tell Codex:

```text
I restored the repo on the new computer at ~/Desktop/Mpail2-OmniReset-codex-. Please verify environment and continue from the S190/S192/IQ-Learn + MPAIL debugging context.
```

5. Start with verification commands, not a new algorithm change.

## If The External Drive Is Mounted Read-Only

Check:

```bash
findmnt -rno TARGET,SOURCE,FSTYPE,OPTIONS | rg 'MPAIL_BACKUP|/dev/sd'
```

If the ext4 partition is read-only, repair it:

```bash
sudo umount /dev/sda1
sudo e2fsck -f /dev/sda1
sudo mkdir -p /media/$USER/MPAIL_BACKUP
sudo mount -o rw /dev/sda1 /media/$USER/MPAIL_BACKUP
```

Replace `/dev/sda1` if `lsblk` shows a different device name on the new computer.

## Safe Eject

Before unplugging:

```bash
sync
sudo umount /media/$USER/MPAIL_BACKUP
sudo udisksctl power-off -b /dev/sda
```

If the drive has the tiny `UEFI_NTFS` partition mounted too, unmount it before power-off:

```bash
sudo umount /media/$USER/UEFI_NTFS
sudo udisksctl power-off -b /dev/sda
```

Again, replace `/dev/sda` if `lsblk` shows a different drive name.
