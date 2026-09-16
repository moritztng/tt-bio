#!/bin/bash
# The same cell again, with the card ARC clock recorded per fold. Same wait-for-an-empty-box
# discipline as run_cell_clean2.sh; the only new thing in the record is AICLK.
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
for i in $(seq 1 12); do
  [ "$(holders)" = "0" ] && { echo "box free after $((i*15))s"; break; }
  sleep 15
done
[ "$(holders)" = "0" ] || { echo "STILL-COTENANTED"; echo CELLCLOCKDONE; exit 75; }
/home/ttuser/.coworker/scripts/benchlock.sh b2z2-wave2-cell-recheck-clock -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:b2z2-wave2-cell-recheck PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_cell_recheck/cell_now.py \
    --out perf/b2z2_cell_recheck/out/cell_main_qb2c1_clock.json --reps 6 --warm 1 \
    --clock "tenstorrent!1"
echo "RC=$?"
echo CELLCLOCKDONE
