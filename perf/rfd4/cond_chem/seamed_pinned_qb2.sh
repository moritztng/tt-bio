#!/bin/bash
# nucleic_acid is the one family that needs BOTH corrections at once, and neither alone is enough:
#   - pin the expandable segment to upstream's realized 126 (perf/rfd4/cond_chem/pin_segment.py),
#     or our batch is 153 tokens against upstream's 145 and the two runs design different problems;
#   - transplant upstream's featurizer seams, which is what took glycan_assembly from 6.256 A to
#     0.166 A at the same seed.
# The seam dump is 145 tokens wide, so it only fits the PINNED input. Unpinned it raises
#   IndexError: The shape of the mask [1, 153] does not match the indexed tensor [1, 145]
# which is the same shape mismatch read from the other end.
#   TT_VISIBLE_DEVICES=<card> perf/rfd4/cond_chem/seamed_pinned_qb2.sh <out-root>
set -uo pipefail
OUT=${1:?out-root}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
RUNS=$HOME/rfd4-cond-chem-runs
PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
SEED=1

cd "$WT" || exit 1
for try in 1 2 3 4 5 6; do
  if ls "$OUT/nucleic_acid/seed$SEED"/*_generated.cif >/dev/null 2>&1; then
    echo "=== already designed"; break
  fi
  echo "=== attempt $try start $(date -u +%FT%TZ)"
  PYTHONPATH=$WT timeout 3600 "$PY" perf/rfd4/e2e/design.py \
    --cif "$RUNS/inputs/dna_binder_len126.cif" \
    --recipe medium/front_loaded --weights nvidia/RFD4-Proteina \
    --out "$OUT/nucleic_acid/seed$SEED" --seed "$SEED" --designs 2 --rng upstream \
    --upstream-seams "$RUNS/seams/nucleic_acid/seed$SEED/seams.pt"
  echo "=== attempt $try rc=$? end $(date -u +%FT%TZ)"
  sleep 180
done
echo "### seamed+pinned done $(date -u +%FT%TZ)"
