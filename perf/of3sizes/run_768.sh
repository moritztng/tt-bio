#!/bin/bash
WT=/home/ttuser/.coworker/wt/openfold3-sizes-perf
cd $WT
exec /home/ttuser/.coworker/scripts/benchlock.sh openfold3-sizes-perf -- env \
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:openfold3-sizes-perf \
  PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt \
  python3 -u perf/other512/fold_ab_multi.py --model openfold3 --sizes 768 \
    --arms on,nofuse,on --out perf/of3sizes/ab_768_qb1c0.json
