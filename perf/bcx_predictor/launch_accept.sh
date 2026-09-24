#!/bin/bash
# Four more paired device trajectories, on the seeds the reference arms run. The loop now
# clears three of BindCraft 2's four gradient stages, so more trajectories are productive:
# the campaign needs an ACCEPTED design before chip-seconds per accepted design exists.
set -u
WT=/home/ttuser/.coworker/wt/bcx-predictor
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-predictor
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
for SEED in 200 300 400 0; do
  OUT=$WT/perf/bcx_predictor/runs/accept_seed$SEED
  mkdir -p "$OUT"
  /home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
    --arm device --trajectories 1 --seed "$SEED" --bucket 0 --out "$OUT" \
    >>"$OUT/run.log" 2>&1
  echo "seed $SEED rc=$? $(date -u +%FT%TZ)" >> "$WT/perf/bcx_predictor/runs/accept_progress.log"
done
