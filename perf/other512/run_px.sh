#!/bin/bash
# protenix-v2 512 aa: the shipped (kt, nt) key vs main's nt-only key, alternating, 3 rounds each.
WT=/home/ttuser/.coworker/wt/other-models-512aa-protenix-regression
cd $WT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:other-models-512aa-protenix-regression PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh other-models-512aa-protenix-regression -- \
  $PY -u perf/other512/fold_ab_multi.py --model protenix-v2 --sizes 512 \
      --arms on,oldkey,on,oldkey,on,oldkey --out perf/other512/ab_px_rekey_512.json
echo "RC=$?"
