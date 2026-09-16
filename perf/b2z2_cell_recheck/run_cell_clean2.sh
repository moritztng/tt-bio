#!/bin/bash
# Second attempt at the clean cell on card 1. The first waited 1230s, got the box, then lost it
# three folds in to a sibling worker that opened card 0 without benchlock. This one waits again,
# and cell_now.py refuses to start with a foreign holder and records the holder list per fold, so
# a session that goes co-tenanted halfway says so in its own record rather than averaging it in.
set -u
WT=/home/ttuser/.coworker/wt/b2z2-wave2-cell-recheck
cd "$WT" || exit 1
holders() {
  local n=0 p
  for p in $(ls /proc | grep -E "^[0-9]+$"); do
    [ "$p" = "$$" ] && continue
    if ls -l /proc/$p/fd 2>/dev/null | grep -q tenstorrent; then n=$((n+1)); fi
  done
  echo "$n"
}
for i in $(seq 1 50); do
  h=$(holders)
  if [ "$h" = "0" ]; then echo "box free at $(date -u +%H:%M:%S) after $((i*30))s"; break; fi
  [ $((i % 4)) -eq 1 ] && echo "$(date -u +%H:%M:%S) waiting: $h foreign device holder(s)"
  sleep 30
done
[ "$(holders)" = "0" ] || { echo "STILL-COTENANTED after 25min"; echo CELLCLEAN2DONE; exit 75; }
/home/ttuser/.coworker/scripts/benchlock.sh b2z2-wave2-cell-recheck-clean2 -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:b2z2-wave2-cell-recheck PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_cell_recheck/cell_now.py \
    --out perf/b2z2_cell_recheck/out/cell_main_qb2c1_clean2.json --reps 10 --warm 2
echo "RC=$?"
echo CELLCLEAN2DONE
