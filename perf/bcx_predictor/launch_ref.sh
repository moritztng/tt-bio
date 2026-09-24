#!/bin/bash
# The reference arm: BindCraft 2 unmodified, its own JAX predictor, CPU, monomer pool,
# length_bucket_size 1 so the complex is unpadded and the fold is unmasked -- the same
# configuration the device arm must run, since tt-bio's AF2 trunk asserts an all-ones mask.
# Rooted in this worker's own worktree (worker_prompt rule 5).
set -u
WT=/home/ttuser/.coworker/wt/bcx-predictor
cd "$WT" || exit 1
OUT="$WT/perf/bcx_predictor/runs/reference_qb2_b1"
mkdir -p "$OUT"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec /home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
  --arm reference --trajectories 10 --seed 0 --bucket 1 --out "$OUT" \
  >>"$OUT/run.log" 2>&1
