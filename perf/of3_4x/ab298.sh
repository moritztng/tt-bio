#!/usr/bin/env bash
# §7 step 7: the 298 aa control, RMSD only. cdk2x2_512 is a chimera whose hinge saturates RMSD near
# 8 A at 512 aa regardless of cause, so a large 512 aa number is not diagnostic on its own.
# Not benchlocked: no timing claim is made from this, so the box goes back to the other tasks.
set -u
PY=/home/ttuser/tt-bio-dev/env/bin/python3
WT=/home/ttuser/.coworker/wt/openfold3-to-4x
BASE=/tmp/of3_base
OUT=$WT/perf/of3_4x/ab298
mkdir -p "$OUT"
for arm in on P; do
  [ "$arm" = on ] && root=$BASE || root=$WT
  echo "=== 298 arm $arm ($root) $(date -Is) ==="
  (cd "$root" && TT_VISIBLE_DEVICES=3 \
    TT_BIO_LEASE_HOLDER=worker:openfold3-to-4x \
    PYTHONPATH="$root" "$PY" perf/of3deep/decomp.py --size 298 \
      --out "$OUT/${arm}.json" --keep-cif "$OUT/cif_${arm}") \
    2>&1 | grep -E "^  cold|^=== run|Error|Traceback|assert"
  echo "--- exit ${PIPESTATUS[0]} 298 arm $arm $(date -Is)"
done
echo "ALL 298 DONE $(date -Is)"
