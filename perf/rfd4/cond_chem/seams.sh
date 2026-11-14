#!/bin/bash
# Upstream seam dumps for the three chemistry families whose design cannot be seed-matched.
#   perf/rfd4/cond_chem/seams.sh <seed>
# rfd4-pair showed the gap is upstream's featurizer: chain_iid, molecule_iid and the
# internal-coordinate frame come off the process-global streams, so they are not a function of
# the design seed and must be transplanted. This dumps them, per family, at the same seed the
# device design runs. rna is already seed-matched and is not here.
set -uo pipefail
SEED=${1:-1}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
ROOT=/home/moritz/rfd4-ref-env; SRC=$ROOT/src/RFD4-Proteina
export PROJECT_ROOT=$SRC HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export PYTHONPATH=$WT:$WT/perf/rfd4/e2e:$WT/perf/rfd4/ref
cd "$SRC" || exit 1

FAMILIES=(
  "small_molecule  $SRC/data/benchmarks/small_molecule_binder/sm_binder  SAM            medium/midpeak"
  "nucleic_acid    $SRC/data/benchmarks/dna_binder/dna_binder_denovo_val 7m5w_structure_context medium/front_loaded"
  "glycan_assembly /home/moritz/rfd4-cond-chem-runs/inputs/glycan-annotated glycan_binder thinking_medium"
)
for row in "${FAMILIES[@]}"; do
  read -r NAME INPUTS SUBSET RECYCLE <<<"$row"
  OUT=/home/moritz/rfd4-cond-chem-runs/seams/$NAME/seed$SEED
  [ -s "$OUT/seams.pt" ] && { echo "=== $NAME seed $SEED already dumped"; continue; }
  mkdir -p "$OUT"
  echo "=== $NAME seed $SEED start $(date -u +%FT%TZ)"
  nice -n 19 "$ROOT/venv/bin/python" "$WT/perf/rfd4/e2e/pair_upstream_dump.py" \
    --seed "$SEED" --steps 1 --out "$OUT" \
    "inputs=$INPUTS" "subset_to_examples=$SUBSET" "recycle_strategy=$RECYCLE" \
    "out_dir=$OUT/designs" "seed=$SEED" num_replicates=1 batch_size=1 '~metrics' compile_lengths=null
  echo "=== $NAME seed $SEED rc=$? end $(date -u +%FT%TZ)"
done
