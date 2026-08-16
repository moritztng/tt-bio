#!/usr/bin/env bash
# Step 1: the BD arm at 512 aa, low load. Replaces the -0.555 s bracketed estimate with a
# measurement. `ship` in the same process is the A/A floor and the reference for the delta.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=5400
/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200-p2 -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esm3p4land/fold_ab.py \
  --model esmfold2 --size 512 --rounds 3 --arms ship,B,D,BD \
  --out perf/esmbeat/p2_bd_512_c0.json
echo "EXIT=$?"
