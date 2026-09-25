#!/bin/bash
# The device arm: BindCraft 2's campaign with tt-bio's Evoformer on card 3, one trajectory.
set -u
WT=/home/ttuser/.coworker/wt/bcx-predictor
OUT=$WT/perf/bcx_predictor/runs/device_qb2
mkdir -p "$OUT"
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-predictor
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec /home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
  --arm device --trajectories 1 --seed 0 --bucket 1 --out "$OUT" >>"$OUT/run.log" 2>&1
