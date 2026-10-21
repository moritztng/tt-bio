#!/bin/bash
# The four non-protein condition families, designed on one Blackhole chip.
#   TT_VISIBLE_DEVICES=<card> perf/rfd4/cond_chem/designs_qb2.sh <out-root> [seed...]
# Two designs per family per seed: the first pays kernel compilation, the second is warm.
# Each family runs at the recipe RFD4 prescribes for its task, and `rng=upstream` so the
# trajectory pairs with `upstream_design.sh --reseed-at-sampler` at the same seed.
set -uo pipefail
OUT=${1:?out-root}; shift
SEEDS=${*:-0}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
IN=$HOME/.coworker/state/rfd4/inputs
PY=${PY:-$HOME/tt-bio-dev/env/bin/python}

# family                 annotated CIF            recipe
FAMILIES=(
  "small_molecule        $IN/sam_binder.cif       medium/midpeak"
  "nucleic_acid          $IN/dna_binder.cif       medium/front_loaded"
  "nucleic_ss            $IN/rna_pseudoknot.cif   medium/front_loaded"
  "glycan_assembly       $IN/glycan_binder.cif    thinking_medium"
)

cd "$WT" || exit 1
for SEED in $SEEDS; do
  for row in "${FAMILIES[@]}"; do
    read -r NAME CIF RECIPE <<<"$row"
    [ -f "$CIF" ] || { echo "$NAME: missing $CIF"; continue; }
    echo "=== $NAME seed $SEED recipe $RECIPE start $(date -u +%FT%TZ)"
    PYTHONPATH=$WT timeout 3600 "$PY" perf/rfd4/e2e/design.py \
      --cif "$CIF" --recipe "$RECIPE" --weights nvidia/RFD4-Proteina \
      --out "$OUT/$NAME/seed$SEED" --seed "$SEED" --designs 2 --rng upstream
    echo "=== $NAME seed $SEED rc=$? end $(date -u +%FT%TZ)"
  done
done
