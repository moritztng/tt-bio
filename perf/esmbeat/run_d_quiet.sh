#!/usr/bin/env bash
set -u
cd /home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200
export PYTHONPATH=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=5400
/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200 -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esm3p4land/fold_ab.py \
  --model esmfold2 --size 512 --rounds 3 --arms ship,D,ABD \
  --out perf/esmbeat/ab_d_512_c0_quiet.json
echo "EXIT=$?"
