#!/bin/bash
WT=/home/ttuser/.coworker/wt/other-models-512aa-perf
cd $WT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:other-models-512aa-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
for M in boltz2 openfold3; do
  echo "=== $M 512 ==="
  $PY -u perf/z_flip_land/census_sweep.py --model $M \
     --targets perf/size512/fixtures/cdk2x2_512.yaml \
     --out perf/other512/repro_${M}_512.json 2>&1 | tail -80
  echo "=== $M rc=$? ==="
done
