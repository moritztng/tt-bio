#!/bin/bash
# Re-take the cell on card 1 once nothing else on this host holds a Tenstorrent device.
# The first attempt ran with b2z2-conf-readback-bh-fix folding on card 0, the board partner of
# card 1, for every one of its ten folds; this box reads ~14.7 s idle and ~21.5 s like that.
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
for i in $(seq 1 90); do
  h=$(holders)
  if [ "$h" = "0" ]; then echo "box free after $((i*20))s"; break; fi
  echo "waiting: $h foreign device holder(s)"
  sleep 20
done
[ "$(holders)" = "0" ] || { echo "STILL-COTENANTED"; echo CELLRECHECKCLEANDONE; exit 75; }
/home/ttuser/.coworker/scripts/benchlock.sh b2z2-wave2-cell-recheck-clean -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:b2z2-wave2-cell-recheck PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_cell_recheck/cell_now.py \
    --out perf/b2z2_cell_recheck/out/cell_main_qb2c1_clean.json --reps 10 --warm 2
echo "RC=$?"
echo CELLRECHECKCLEANDONE
