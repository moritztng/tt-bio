#!/bin/bash
# Load-sensitivity control, card 3, p300c, pinned 1350 MHz.
#
# b2_old_s1 came back 35.362 s on this card against a 22.01 s prediction, at loadavg 18.31 with
# 4 of 4 folds co-tenanted. The question that decides whether that number is garbage or a finding:
# does host-CPU load inflate the OLD tree more than the NEW one? The old tree is the less
# optimized, more dispatch-bound arm, so host starvation should hit it harder -- but "should" is
# not a measurement.
#
# Deliberately run UNDER load. Sandwiched new/old/new so load drift across the session is visible
# in the arm that brackets it. Runs WITHOUT taking benchlock because this worker already holds it
# (pid 1178270); nobody else is folding.
set -u
NEW=/home/ttuser/pvx_qb2/new
OLD=/home/ttuser/pvx_qb2/old
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_qb2/out4
mkdir -p "$OUT"
one() {
  local tree=$1 tag=$2 reps=$3
  echo "=== $(date -u +%H:%M:%SZ) $tag tree=$tree reps=$reps loadavg=$(cut -d" " -f1-3 /proc/loadavg) ==="
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
      PYTHONPATH="$tree" \
    "$PY" -u "$tree/perf/pvx_baseline/cell.py" --model boltz2 --reps "$reps" --clock 1350 \
      --out "$OUT/$tag.json" --tag "$tag" 2>&1 | tail -4
  echo "RC=${PIPESTATUS[0]} $tag"
}
one "$NEW" b2_load_new_a 3
one "$OLD" b2_load_old_a 3
one "$NEW" b2_load_new_b 3
echo "LOADSENSDONE $(date -u +%FT%TZ) loadavg=$(cut -d" " -f1-3 /proc/loadavg)"
