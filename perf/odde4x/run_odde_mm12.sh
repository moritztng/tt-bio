#!/bin/bash
# `mm12` at 512 aa on THIS tree, so its headline is measured against main-with-ge (96.785 s) rather
# than against the predecessor's pre-ge `on`. Arms on,mm12,on, one process.
WT=/home/ttuser/.coworker/wt/opendde-to-4x
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-to-4x -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 512 \
      --arms on,mm12,on --out perf/odde4x/ab_opendde_512_mm12.json
echo "RC=$?"
