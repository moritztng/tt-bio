#!/bin/bash
# The shipped path: gated_move now defaults True at Protenix.load_from_checkpoint, so the
# 'on' arm is what a user gets. --instrument splits every fold into featurize / model.fold /
# CIF write, because the page's GPU leg times model.fold only.
WT=/home/ttuser/.coworker/wt/protenix-v2-to-4x-per-dollar
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-to-4x-per-dollar PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh protenix-v2-to-4x-per-dollar -- \
  $PY -u perf/px4pd/e6_ab512.py --wall 0 --instrument 1 --arms off,ontr,off,ontr \
      --out perf/px4pd/e6_integrated_qb2c1.json
echo "RC=$?"
