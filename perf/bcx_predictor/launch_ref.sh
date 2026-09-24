#!/bin/bash
# The reference arm: BindCraft 2 unmodified, its own JAX predictor, CPU, monomer pool.
# Rooted in this worker's own worktree (worker_prompt rule 5) so fleet hygiene cannot
# delete the run out from under it.
set -u
WT=/home/ttuser/.coworker/wt/bcx-predictor
cd "$WT" || exit 1
OUT="$WT/perf/bcx_predictor/runs/reference_qb2"
mkdir -p "$OUT"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
exec /home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
  --arm reference --trajectories 10 --seed 0 --out "$OUT" \
  >>"$OUT/run.log" 2>&1
