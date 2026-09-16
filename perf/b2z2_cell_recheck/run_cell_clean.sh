#!/bin/bash
# Take the cell on card 1 as soon as nothing else on this host holds a Tenstorrent device.
# Detached and patient by design: the first attempt was spoiled by a sibling worker folding on
# card 0, the board partner, and this box reads ~14.7 s idle against ~21.5 s like that.
set -u
WT=/home/ttuser/.coworker/wt/b2z2-wave2-cell-recheck
cd "$WT" || exit 1
holders() {
  local n=0 p
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    [ "$p" = "$$" ] && continue
    if ls -l /proc/$p/fd 2>/dev/null | grep -q tenstorrent; then n=$((n+1)); fi
  done
  echo "$n"
}
for i in $(seq 1 360); do
  h=$(holders)
  if [ "$h" = "0" ]; then echo "box free at $(date -u +%H:%M:%S) after $((i*30))s"; break; fi
  [ $((i % 10)) -eq 1 ] && echo "$(date -u +%H:%M:%S) waiting: $h foreign device holder(s)"
  sleep 30
done
[ "$(holders)" = "0" ] || { echo "STILL-COTENANTED after 3h"; echo CELLRECHECKCLEANDONE; exit 75; }
/home/ttuser/.coworker/scripts/benchlock.sh b2z2-wave2-cell-recheck-clean -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:b2z2-wave2-cell-recheck PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_cell_recheck/cell_now.py \
    --out perf/b2z2_cell_recheck/out/cell_main_qb2c1_clean.json --reps 10 --warm 2
echo "RC=$?"
echo CELLRECHECKCLEANDONE
