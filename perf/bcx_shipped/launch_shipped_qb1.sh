#!/bin/bash
# Reference arm 3 -- the SHIPPED PD-L1 configuration on qb1, CPU, seed 2.
#
# Same configuration as arm 1 (pc, seed 0) and the adopted arm 2 (qb2, seed 1): BindCraft 2s
# own resolver gives examples/pdl1.json five multimer_v3 design models and two monomer
# validation models, --bucket 0 leaves length_bucket_size at BindCraft 2s own default of 32,
# and nothing else is set but the seed and where the project lands.
#
# Why qb1, and why this workload. A reference arm is BindCraft 2s own JAX on CPU, so it opens
# NO device: it cannot trigger the device-open hang that took qb1 down twice on 25 Sep. If the
# box dies we lose this trajectorys remaining wall time and no data -- completed trajectories
# are written under the arm directory as they finish. The campaign needs reference
# trajectories, not device ones: at 2 completed the per-stage clause rejects two identical
# programs two thirds of the time, and permutation resolution triples per added reference
# sample.
#
# Rooted in this rows OWN directory on qb1 (worker_prompt rule 5), never in bcx-exacts
# qb1tree and never in /home/ttuser/bcx_e2e/bc2, which is bcx-seeds banked tree and is read
# only to everyone.
set -u
WT=/home/ttuser/bcx_shipped_qb1/shiptree
cd "$WT" || exit 1
OUT=/home/ttuser/bcx_shipped_qb1/ref_s2
mkdir -p "$OUT"
# qb1 has 32 cores and bcx-exacts three device arms already draw about 10 of them.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export PYTHONPATH=/home/ttuser/bcx_e2e/bc2
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u "$WT/perf/bcx_predictor/run_arm.py" \
  --arm reference --shipped --trajectories 10 --seed 2 --bucket 0 \
  --params /home/ttuser/bcx_e2e/af2_params \
  --settings /home/ttuser/bcx_e2e/bc2/examples/pdl1.json \
  --out "$OUT" >>/home/ttuser/bcx_shipped_qb1/ref_s2.log 2>&1
