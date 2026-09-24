#!/bin/bash
# The device acceptance arm: BindCraft 2's campaign with tt-bio's Evoformer on card 3,
# one trajectory per seed, on the SAME seeds as the five reference streams so the arms
# are paired. The trajectory-name hash is the check that the pairing held.
set -u
WT=/home/ttuser/.coworker/wt/bcx-predictor
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-predictor
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
for SEED in 0 100 200 300 400; do
  OUT=$WT/perf/bcx_predictor/runs/device_qb2_seed$SEED
  mkdir -p "$OUT"
  /home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
    --arm device --trajectories 1 --seed "$SEED" --bucket 1 --out "$OUT" \
    >>"$OUT/run.log" 2>&1
  echo "seed $SEED rc=$? $(date -u +%FT%TZ)" >> "$WT/perf/bcx_predictor/runs/device_progress.log"
done
