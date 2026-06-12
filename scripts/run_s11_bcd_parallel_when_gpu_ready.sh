#!/usr/bin/env bash
set -euo pipefail
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

launcher_root="$PWD/logs/s11_bcd_gpu_wait_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$launcher_root"

echo "[gpu-wait] $(date): waiting for nvidia-smi to report a usable GPU" | tee "$launcher_root/status.log"

while true; do
  if nvidia-smi > "$launcher_root/nvidia-smi.last" 2>&1; then
    echo "[gpu-wait] $(date): GPU is available; launching B/C/D parallel sweep" | tee -a "$launcher_root/status.log"
    exec bash scripts/run_s11_bcd_parallel_300.sh
  fi

  echo "[gpu-wait] $(date): GPU unavailable; retrying in 60s" | tee -a "$launcher_root/status.log"
  sleep 60
done
