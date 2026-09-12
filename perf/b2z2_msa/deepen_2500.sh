#!/bin/sh
# The 2500-row alignment the deep A/B was folded against. Repetition alone does not deepen an
# alignment -- the Boltz-2 featurizer deduplicates -- so each copy carries one substituted column.
# Regenerated rather than committed: 2500 x 512 residues is 2.6 MB of synthetic rows.
exec python3 "$(dirname "$0")/../b2z_work_removal/deepen_a3m.py" \
  --src "$(dirname "$0")/../size512/fixtures/cdk2x2_512.a3m" \
  --out "${1:?usage: deepen_2500.sh <out.a3m>}" --depth 2500
