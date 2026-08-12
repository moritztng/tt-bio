#!/bin/bash
WT=/home/ttuser/.coworker/wt/other-models-512aa-perf
cd $WT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:other-models-512aa-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh other-models-512aa-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 512 \
      --arms on,nonewmm,on,nonewmm --out perf/other512/ab_b2_rekey_512.json
echo "RC=$?"
