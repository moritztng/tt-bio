#!/bin/sh
# The one fold owed by esmfold2-sizes-perf 16.5: armA (row block OFF) at 768 aa on qb2 s 110-core
# grid, against armAC (= shipped at 768) in the same process. Answers whether the 3.0x unblocked
# pathology measured on qb1 s 130-core grid is grid-specific or size-specific.
set -e
cd /home/ttuser/.coworker/wt/esmfold2-qb2-768-grid-check
exec env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:esmfold2-qb2-768-grid-check \
  PYTHONPATH=/home/ttuser/.coworker/wt/esmfold2-qb2-768-grid-check \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/esm2land/fold_ab.py \
    --model esmfold2 --size 768 --arms armAC,armA --rounds 3 \
    --out perf/esm2sizes/fold_768_armA_qb2c1.json
