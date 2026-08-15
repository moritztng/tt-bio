#!/bin/bash
WT=/home/ttuser/.coworker/wt/protenix-v2-to-4x-per-dollar
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-to-4x-per-dollar PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh protenix-v2-to-4x-per-dollar -- \
  $PY -u perf/px4pd/e6_ab512.py --wall 0 --arms off,on,off,on \
      --out perf/px4pd/e6_ship_qb2c1.json
echo "RC=$?"
