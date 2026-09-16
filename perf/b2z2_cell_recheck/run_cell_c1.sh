#!/bin/bash
set -u
WT=/home/ttuser/.coworker/wt/b2z2-wave2-cell-recheck
cd "$WT" || exit 1
mkdir -p perf/b2z2_cell_recheck/out
/home/ttuser/.coworker/scripts/benchlock.sh b2z2-wave2-cell-recheck -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:b2z2-wave2-cell-recheck PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_cell_recheck/cell_now.py \
    --out perf/b2z2_cell_recheck/out/cell_main_qb2c1.json --reps 10 --warm 2
echo "RC=$?"
echo CELLRECHECKDONE
