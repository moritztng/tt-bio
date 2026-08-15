#!/bin/bash
# Where the two levers' gates open and shut. off = main default (E6 off, headroom 2.5);
# ontr = both levers, which is what merging this branch ships. Each size is its own process
# and its own device context. 768 gets two arms rather than four -- its delta is single-shot
# and labelled as such.
WT=/home/ttuser/.coworker/wt/protenix-v2-to-4x-per-dollar
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-to-4x-per-dollar PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
for s in 256 298 640 768; do
  arms=off,ontr,off,ontr
  [ $s = 768 ] && arms=off,ontr
  echo "===== size $s arms $arms ====="
  /home/ttuser/.coworker/scripts/benchlock.sh protenix-v2-to-4x-per-dollar -- \
    $PY -u perf/px4pd/e6_ab512.py --wall 0 --size $s --arms $arms \
        --out perf/px4pd/sweep_${s}_qb2c1.json
  echo "size $s RC=$?"
done
echo ALLDONE
