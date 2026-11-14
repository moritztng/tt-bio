#!/bin/bash
# The three families whose trajectories did not pair, re-designed at seed 1 with upstream's own
# featurizer seams transplanted (perf/rfd4/cond_chem/seams.sh wrote them on pc).
#   TT_VISIBLE_DEVICES=<card> perf/rfd4/cond_chem/seamed_qb2.sh <out-root> <seam-root>
# Idempotent the same way designs_qb2.sh is: a family already on disk is skipped.
set -uo pipefail
OUT=${1:?out-root}
SEAMS=${2:?seam-root}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
IN=$HOME/.coworker/state/rfd4/inputs
PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
SEED=1

FAMILIES=(
  "small_molecule   $IN/sam_binder.cif      medium/midpeak"
  "nucleic_acid     $IN/dna_binder.cif      medium/front_loaded"
  "glycan_assembly  $IN/glycan_binder.cif   thinking_medium"
)

cd "$WT" || exit 1
for row in "${FAMILIES[@]}"; do
  read -r NAME CIF RECIPE <<<"$row"
  SEAM=$SEAMS/$NAME/seed$SEED/seams.pt
  [ -f "$CIF" ]  || { echo "$NAME: missing $CIF"; continue; }
  [ -f "$SEAM" ] || { echo "$NAME: missing $SEAM"; continue; }
  if ls "$OUT/$NAME/seed$SEED"/*_generated.cif >/dev/null 2>&1; then
    echo "=== $NAME seed $SEED already designed, skipping"; continue
  fi
  echo "=== $NAME seed $SEED recipe $RECIPE seamed start $(date -u +%FT%TZ)"
  PYTHONPATH=$WT timeout 3600 "$PY" perf/rfd4/e2e/design.py \
    --cif "$CIF" --recipe "$RECIPE" --weights nvidia/RFD4-Proteina \
    --out "$OUT/$NAME/seed$SEED" --seed "$SEED" --designs 2 --rng upstream \
    --upstream-seams "$SEAM"
  echo "=== $NAME seed $SEED rc=$? end $(date -u +%FT%TZ)"
done
