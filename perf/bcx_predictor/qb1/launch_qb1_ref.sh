#!/bin/bash
# Four independent reference trajectories on qb1, 8 threads each across its 32 cores.
# Seeds do not overlap the qb2 arm's (campaign_seed 0). Everything runs from tmpfs:
# qb1's root disk has 1.2 G free, and /dev/shm is RAM-backed with 244 G.
#
# tmpfs is VOLATILE. These outputs do not survive a qb1 reboot and must be copied to the
# worktree on qb2 before they are quoted.
set -u
R=/dev/shm/bcx-ref
export BCX_BC2=$R/bc2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
for SEED in 100 200 300 400; do
  OUT=$R/runs/reference_qb1_seed$SEED
  mkdir -p "$OUT"
  setsid nohup "$R/venv/bin/python" "$R/harness/run_arm.py" \
    --arm reference --trajectories 3 --seed "$SEED" --bucket 1 \
    --params "$R/af2_params" --out "$OUT" >>"$OUT/run.log" 2>&1 &
  disown
  sleep 2
done
sleep 3
pgrep -af "run_arm.py" | wc -l
