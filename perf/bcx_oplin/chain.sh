#!/bin/bash
# Every predict model through fold_ab.py, one process each, on card 0.
cd /home/ttuser/.coworker/wt/bcx-oplin
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-oplin
for m in "$@"; do
  ~/tt-bio-dev/env/bin/python perf/bcx_oplin/fold_ab.py --model $m --out perf/bcx_oplin/folds/${m}_512 > perf/bcx_oplin/folds/${m}_512.log 2>&1
  echo "$m rc=$? $(date -u +%FT%TZ)" >> perf/bcx_oplin/folds/chain.log
done
