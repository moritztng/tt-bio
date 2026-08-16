#!/usr/bin/env bash
set -u
cd /home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200
export PYTHONPATH=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200
export ESM_ROOT=/home/ttuser/esm BENCHLOCK_WAIT_S=5400
/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200 -- \
  timeout -s KILL 420 /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esmbeat/s_d_writesplit.py \
  --arms generic,nowrite,dualnoc --rounds 3 --out perf/esmbeat/s_d_dualnoc.json
echo "EXIT=$?"
