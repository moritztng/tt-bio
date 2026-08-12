#!/bin/bash
WT=/home/ttuser/.coworker/wt/other-models-512aa-perf
cd $WT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:other-models-512aa-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh other-models-512aa-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model openfold3 --sizes 512 \
      --arms on,nofp32hifi,nofp32,on --out perf/other512/ab_of3_hifi_512.json
echo "RC=$?"
