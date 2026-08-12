#!/bin/bash
WT=/home/ttuser/.coworker/wt/opendde-to-4x
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x PYTHONPATH=$WT
export BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-to-4x -- \
  $PY -u perf/odde4x/screen1.py --out perf/odde4x/screen1.json
echo "RC=$?"
