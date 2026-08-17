#!/usr/bin/env bash
# The qb2 page-cell remeasurement. The published 31.448 s cell is a qb2 card-0 `ship` median over
# 4 rounds (perf/esmbeat/p2_ship_512_c0_postmerge.json); this reruns that exact protocol on the
# tree that carries E, C-in, F and G. `ship` names no lever, so it measures the module default
# and nothing else -- the only arm that catches a default that did not land.
#
# Run 2 (--arms ship,off) is the calibration control. qb2 measured `off` at 31.944 s in p2; if
# today reads within ~0.15 s of that, the box is the same box that produced the published cell and
# the absolute wall-clock comparison against the 29.024 s bar is sound.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-qb2-remeasure
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-qb2-remeasure
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=600
PY=/home/ttuser/tt-bio-dev/env/bin/python3

/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200-qb2-remeasure -- \
  "$PY" -u perf/esm3p4land/fold_ab.py \
  --model esmfold2 --size 512 --rounds 4 --arms ship \
  --out perf/esmbeat/qb2_ship_512_c0.json
echo "RUN1_EXIT=$?"

/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200-qb2-remeasure -- \
  "$PY" -u perf/esm3p4land/fold_ab.py \
  --model esmfold2 --size 512 --rounds 4 --arms ship,off \
  --out perf/esmbeat/qb2_shipoff_512_c0.json
echo "RUN2_EXIT=$?"
