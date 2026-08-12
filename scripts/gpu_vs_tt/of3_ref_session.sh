#!/usr/bin/env bash
# The whole rented-GPU leg of openfold3-fused-sdpa-gpu-reference-check: ten OpenFold3
# reference folds (298 aa and 512 aa, seeds 0-4), each gated on its own structure, each
# harvested to /root/results/of3ref. Nothing else runs on the box.
#
# Five seeds per size is not decoration: a single reference fold gives one basin and no scale
# to read a 0.27 A arm-to-arm margin against. The seed spread IS the scale.
#
#   bash of3_ref_session.sh                  # both sizes, seeds 0-4
#   SIZES=298 SEEDS="0 1" bash of3_ref_session.sh
set -uo pipefail          # NOT -e: one bad fold must not cost the other nine
cd "$(dirname "$0")"
HERE=$(pwd)
R=${R:-/root/results/of3ref}
SIZES=${SIZES:-"298 512"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
PY=${PY:-/root/venv-of3/bin/python3}
PER_FOLD_S=${PER_FOLD_S:-900}
mkdir -p "$R"
START=$(date +%s)

for S in $SIZES; do
  for D in $SEEDS; do
    echo "== ${S} aa seed ${D}: $(date -u +%FT%TZ) ($(( $(date +%s) - START ))s in) =="
    timeout "$PER_FOLD_S" "$PY" of3_ref_one.py --size "$S" --seed "$D" --outdir "$R" \
      > "$R/run_${S}_seed${D}.log" 2>&1
    rc=$?
    echo "   rc=$rc"
    if [ "$rc" -ne 0 ]; then
      tail -25 "$R/run_${S}_seed${D}.log"
      continue
    fi
    python3 "$HERE/gpu5_accuracy_gate.py" "$R/ref_${S}_seed${D}.cif" --expect-residues "$S" \
      | tee "$R/gate_${S}_seed${D}.txt"
  done
done

echo "== session end: $(date -u +%FT%TZ) total=$(( $(date +%s) - START ))s =="
ls -l "$R"/*.cif
