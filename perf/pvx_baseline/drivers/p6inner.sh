#!/bin/bash
# Runs UNDER benchlock. By the time this starts, every OTHER benchlock user is excluded, so any
# load left on the box is a job that does not respect the lock -- today that is of3t's
# bundle_min.py at 657 % CPU. Decide here, inside the lock, and release fast if the box is
# unusable: holding the lock while waiting is what starved two rows this morning.
set -u
NEW=/home/ttuser/pvx_qb2/new
OLD=/home/ttuser/pvx_qb2/old
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_qb2/out3
TAG=${TAG:?}
ENTER_MAX=${ENTER_MAX:-10.0}
l=$(cut -d" " -f1 /proc/loadavg)
echo "    inside the lock at $(date -u +%H:%M:%SZ), loadavg $l, bar $ENTER_MAX"
awk -v a="$l" -v b="$ENTER_MAX" 'BEGIN{exit !(a+0<=b+0)}' || {
  echo "    loadavg $l over $ENTER_MAX, releasing the lock instead of squatting on it"; exit 75; }
for arm in "$OLD:b2_old_$TAG" "$NEW:b2_pairnew_$TAG"; do
  tree=${arm%%:*}; tag=${arm##*:}
  echo "    $(date -u +%H:%M:%SZ) $tag"
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
      PYTHONPATH="$tree" \
    "$PY" -u "$tree/perf/pvx_baseline/cell.py" --model boltz2 --reps 4 --clock 1350 \
      --out "$OUT/$tag.json" --tag "$tag" >>"$OUT/../p6_cells.log" 2>&1 \
    || { echo "    $tag FAILED rc=$?"; exit 1; }
done
echo "    $(date -u +%H:%M:%SZ) pair $TAG folded"
