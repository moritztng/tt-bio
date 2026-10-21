#!/bin/bash
# Upstream CPU fp32 RFD4 design for one condition family, one seed per argument.
#   upstream_design.sh <family> <inputs> <subset> <recycle_strategy> <seed>...
# Paired with tt-bio's DesignEngine rng="upstream": both sides seed the global stream at
# full_simulation's entry, so the two trajectories see the same noise.
#
# <inputs> is a DIRECTORY of STRUCTURE files and <subset> is a file STEM, not a design-problem
# YAML and its top-level key: DesignProblemDataset discovers inputs with iter_structure_files and
# ids each by stem (rfproteina/inference_utils/datasets.py:338), so a .yaml in that directory is
# never seen. Feed it the same annotated CIF tt-bio consumes and both sides read identical bytes.
#
# The four families of this row, as launched:
#   upstream_design.sh sm_binder_sam    $SRC/data/benchmarks/small_molecule_binder/sm_binder \
#                                       SAM                  medium/midpeak      0 1
#   upstream_design.sh dna_binder_7m5w  $SRC/data/benchmarks/dna_binder/dna_binder_denovo_val \
#                                       7m5w_structure_context medium/front_loaded 0 1
#   upstream_design.sh rna_pseudoknot   $SRC/data/benchmarks/rna/pseudoknot \
#                                       Q20                  medium/front_loaded 0 1
#   upstream_design.sh glycan_binder    ~/rfd4-cond-chem-runs/inputs/glycan-annotated \
#                                       glycan_binder        thinking_medium     0 1
# with SRC=/home/moritz/rfd4-ref-env/src/RFD4-Proteina. The glycan family has no upstream
# benchmark input at all, so its annotated CIF is the one this row built from 3AVE chain C.
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
