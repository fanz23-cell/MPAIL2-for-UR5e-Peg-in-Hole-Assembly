#!/usr/bin/env bash
# Serial launcher for S12 a -> b -> c using the actual existing scripts.
# Assumes GPU is already available (no waiting loop).
set -uo pipefail
cd /home/fanz23@netid.washington.edu/Desktop/Mpail2-OmniReset-codex-

launcher_root="$PWD/logs/s12_abc_serial_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$launcher_root"
echo "$launcher_root" | tee "$launcher_root/LAUNCHER_ROOT.txt"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

{
  echo "[$(ts)] starting S12A (h7, raw6, std_floor=0.05)"
  bash scripts/run_s12a_fifo_stdfloor_h7_50.sh > "$launcher_root/S12A.log" 2>&1
  echo "[$(ts)] S12A finished status=$?"

  echo "[$(ts)] starting S12B (h5, raw5, std_floor=0.05)"
  bash scripts/run_s12b_fifo_stdfloor_h5_50.sh > "$launcher_root/S12B.log" 2>&1
  echo "[$(ts)] S12B finished status=$?"

  echo "[$(ts)] starting S12C (h5, raw5, skip_low_std_dims=True)"
  bash scripts/run_s12c_fifo_skiplowstd_h5_50.sh > "$launcher_root/S12C.log" 2>&1
  echo "[$(ts)] S12C finished status=$?"

  echo "[$(ts)] all S12 runs finished"
} | tee "$launcher_root/status.log"
