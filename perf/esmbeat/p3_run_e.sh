#!/usr/bin/env bash
# Lever E at the fold, 512 aa. `ship` carries E (it ships ON), `noE` is the same tree with E
# alone switched off, `off` is the all-levers-off baseline. What this run is for is the CIF sha
# and the plDDT: qb1 is not the page host and its seconds are not a page cell.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=5400
/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200-p3 -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esm3p4land/fold_ab.py \
  --model esmfold2 --size 512 --rounds 3 --arms ship,noE,off \
  --out perf/esmbeat/p3_e_fold_512_c0.json
echo "EXIT=$?"
