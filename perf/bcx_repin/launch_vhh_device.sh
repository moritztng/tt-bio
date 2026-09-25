#!/bin/bash
# bcx-repin leg 3: the device arm on the re-pinned tree, BindCraft 2 unmodified above the trunk.
# examples/pdl1_vhh.json is the example issue #13 reported and the only shipped one PR #17
# changes. Rooted in this worker own worktree (worker_prompt rule 5).
set -u
WT=/home/ttuser/.coworker/wt/bcx-repin
cd "$WT" || exit 1
OUT="$WT/perf/bcx_repin/runs/device_vhh_301efdd_seed0"
mkdir -p "$OUT"
export BCX_BC2=/home/ttuser/bcx_repin/bc2
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:bcx-repin
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec /home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
  --arm device --trajectories 10 --seed 0 --bucket 0 \
  --settings "$BCX_BC2/examples/pdl1_vhh.json" --out "$OUT" \
  >>"$OUT/run.log" 2>&1
