#!/bin/bash
# Same measurement on card 0, the card the published 17.340 s cell was measured on.
set -u
WT=/home/ttuser/.coworker/wt/b2z2-wave2-cell-recheck
cd "$WT" || exit 1
mkdir -p perf/b2z2_cell_recheck/out
/home/ttuser/.coworker/scripts/benchlock.sh b2z2-wave2-cell-recheck-c0 -- \
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=1,0 \
      TT_BIO_LEASE_HOLDER=worker:b2z2-wave2-cell-recheck PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_cell_recheck/cell_now.py \
    --out perf/b2z2_cell_recheck/out/cell_main_qb2c0.json --reps 10 --warm 2
echo "RC=$?"
echo CELLRECHECKC0DONE
