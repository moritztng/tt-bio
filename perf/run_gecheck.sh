#!/bin/bash
# One question: which divisor group does main's shipped search pick at 640 aa?
# Arms on,on: the second `on` is the in-process A/A floor. Cold fold is discarded by the harness.
WT=/home/ttuser/.coworker/wt/opendde-ge-640aa-regression-check
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-ge-640aa-regression-check PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export BENCHLOCK_MAXLOAD=0.6 BENCHLOCK_LOAD_WAIT_S=600
PY=/home/ttuser/tt-bio-dev/env/bin/python3
mkdir -p perf/odde640_gecheck
/home/ttuser/.coworker/scripts/benchlock.sh opendde-ge-640aa-regression-check -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 640 \
      --arms on,on \
      --out perf/odde640_gecheck/ab_opendde_640.json
echo "RC=$?"
echo "=== GECHECK DONE ==="
