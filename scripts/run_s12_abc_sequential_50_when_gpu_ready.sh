#!/usr/bin/env bash
set -euo pipefail
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

launcher_root="$PWD/logs/s12_abc_sequential_50_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$launcher_root"

echo "$launcher_root" | tee "$launcher_root/LAUNCHER_ROOT.txt"
echo "[s12] $(date): waiting for nvidia-smi before running A -> B -> C" | tee "$launcher_root/status.log"

while ! nvidia-smi > "$launcher_root/nvidia-smi.last" 2>&1; do
  echo "[s12] $(date): GPU unavailable; retrying in 60s" | tee -a "$launcher_root/status.log"
  sleep 60
done

echo "[s12] $(date): GPU available; starting S12A" | tee -a "$launcher_root/status.log"
bash scripts/run_s12a_fifo_only_50.sh > "$launcher_root/S12A.log" 2>&1
echo "[s12] $(date): S12A finished; starting S12B" | tee -a "$launcher_root/status.log"

bash scripts/run_s12b_fifo_normfloor_50.sh > "$launcher_root/S12B.log" 2>&1
echo "[s12] $(date): S12B finished; starting S12C" | tee -a "$launcher_root/status.log"

bash scripts/run_s12c_fifo_normfloor_conservative_50.sh > "$launcher_root/S12C.log" 2>&1
echo "[s12] $(date): all S12 runs finished" | tee -a "$launcher_root/status.log"
