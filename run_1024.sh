#!/bin/bash
# CAPACITY probe, deliberately NOT under benchlock: it answers "does 1024 aa fold at all", and the
# co-tenant holds the lock for its own timed anchor. No timing from this run is an A/B.
WT=/home/ttuser/.coworker/wt/openfold3-sizes-perf
cd $WT
exec /home/ttuser/.coworker/scripts/benchlock.sh openfold3-sizes-perf -- env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:openfold3-sizes-perf \
  PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt \
  python3 -u perf/other512/fold_ab_multi.py --model openfold3 --sizes 1024 \
    --arms pre --out perf/of3sizes/pre_1024_qb1c0.json
