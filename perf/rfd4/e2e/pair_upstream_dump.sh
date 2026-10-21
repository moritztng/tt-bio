#!/bin/bash
# Upstream side of the pairing bisect: pdl1-128, seed 0, global stream reseeded at full_simulation's
# entry (the point tt-bio's DesignRng(mode="upstream") starts from), stopped after a couple of steps.
# Dumps x_0, the batch at the first trunk call and the state after each step to seams.pt.
set -uo pipefail
SEED=${1:-0}
STEPS=${2:-2}
WT=/home/moritz/.coworker/wt/rfd4-pair
ROOT=/home/moritz/rfd4-ref-env; SRC=$ROOT/src/RFD4-Proteina
OUT=/home/moritz/rfd4-pair-runs/upstream/${TAG:-seed$SEED}
export PROJECT_ROOT=$SRC HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=6
export PYTHONPATH=$WT/perf/rfd4/e2e:$WT/perf/rfd4/ref
mkdir -p "$OUT"
cd "$SRC"
nice -n 19 "$ROOT/venv/bin/python" "$WT/perf/rfd4/e2e/pair_upstream_dump.py" \
  --seed "$SEED" --steps "$STEPS" --out "$OUT" ${EXTRA:-} \
  "inputs=/home/moritz/.coworker/state/rfd4/inputs/pdl1-128.yaml" "out_dir=$OUT/designs" "seed=$SEED" \
  num_replicates=1 batch_size=1 '~metrics' compile_lengths=null
echo "rc=$? end $(date -u +%FT%TZ)"
