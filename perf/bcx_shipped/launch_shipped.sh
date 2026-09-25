#!/bin/bash
# The SHIPPED PD-L1 configuration, reference arm, CPU, on pc.
#
# `examples/pdl1.json` names no model, so BindCraft 2's own resolver gives it five
# multimer_v3 design models and two monomer validation models. Every other arm in this
# campaign pins design_models=["model_1_ptm"]; --shipped drops that pin, and --bucket 0
# leaves length_bucket_size at BindCraft 2's own default of 32. This is the configuration
# the competition's GO condition names, and nothing here has ever run it.
#
# Reference-only by construction: tt-bio's AF2 trunk is monomer model_1_ptm, so the device
# arm cannot run this pool. No card is opened. Rooted in this worker's own worktree
# (worker_prompt rule 5).
set -u
WT=/home/moritz/.coworker/wt/bcx-shipped
cd "$WT" || exit 1
OUT="$WT/perf/bcx_shipped/runs/shipped_pc"
mkdir -p "$OUT"
# pc is the fleet's dispatch host: 12 cores, 8 to the fold.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export PYTHONPATH=/home/moritz/bcx_shipped/bc2
exec /home/moritz/bcx_tail/venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
  --arm reference --shipped --trajectories 10 --seed 0 --bucket 0 \
  --params /home/moritz/bcx_shipped/af2_params \
  --settings /home/moritz/bcx_shipped/bc2/examples/pdl1.json \
  --out "$OUT" >>"$OUT/run.log" 2>&1
