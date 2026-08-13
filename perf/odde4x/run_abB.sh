#!/bin/bash
# Run B: the capacity check. 640 aa is the hard case -- _PAIR_PROJ_L1_OUT already blocks opendde
# above 640, and the divisor-group search drops to g=6 there. `on` vs `k1def`, one process.
WT=/home/ttuser/.coworker/wt/opendde-to-4x
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-to-4x -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 640 \
      --arms on,k1def --out perf/odde4x/ab_opendde_640.json
echo "RC=$?"
