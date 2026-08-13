#!/bin/bash
# Run A: the fold-level A/B for k1def and k1b at 512 aa, opendde, one process, under benchlock.
# `on` straddles the lever arms so the A/A floor also bounds session drift.
WT=/home/ttuser/.coworker/wt/opendde-to-4x
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-to-4x -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 512 \
      --arms on,k1def,k1b,on --out perf/odde4x/ab_opendde_512.json
echo "RC=$?"
