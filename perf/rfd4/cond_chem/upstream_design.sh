#!/bin/bash
# Upstream CPU fp32 RFD4 design for one condition family, one seed per argument.
#   upstream_design.sh <family> <inputs> <subset> <recycle_strategy> <seed>...
# Paired with tt-bio's DesignEngine rng="upstream": both sides seed the global stream at
# full_simulation's entry, so the two trajectories see the same noise.
set -uo pipefail
FAMILY=$1; INPUTS=$2; SUBSET=$3; RECYCLE=$4; shift 4
ROOT=/home/moritz/rfd4-ref-env; SRC=$ROOT/src/RFD4-Proteina
HARNESS=/home/moritz/.coworker/wt/rfd4-cond-chem/perf/rfd4/ref/capture.py
export PROJECT_ROOT=$SRC HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
cd "$SRC" || exit 1
for SEED in "$@"; do
  OUT=/home/moritz/rfd4-cond-chem-runs/upstream/$FAMILY/seed$SEED; mkdir -p "$OUT"
  echo "$FAMILY seed $SEED start $(date -u +%FT%TZ)"
  nice -n 19 "$ROOT/venv/bin/python" "$HARNESS" --feature "$FAMILY" --seed "$SEED" \
    --reseed-at-sampler --classes _none_ --sampler-steps 0 \
    --out "$OUT/goldens" --records "$OUT/goldens/records.jsonl" \
    "inputs=$INPUTS" "subset_to_examples=$SUBSET" "recycle_strategy=$RECYCLE" \
    "out_dir=$OUT/designs" "seed=$SEED" num_replicates=1 batch_size=1 '~metrics' \
    compile_lengths=null > "$OUT/run.log" 2>&1
  echo "$FAMILY seed $SEED rc=$? end $(date -u +%FT%TZ)"
done
