#!/bin/bash
# Paired interleaved A/B at 298 aa. The first sweep measured BASE on a load-average-18 box and
# C2FIX on a load-average-9 one, which is not a fair ratio: BASE's warm spread was 16%, C2FIX's
# 2%. Here the arms alternate round by round, so a drift in box contention hits both. Score on
# the minimum warm fold across all rounds -- under additive contention noise the minimum is the
# least contaminated estimator -- and report the median alongside it.
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-c2fix-land
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
ROUNDS=${1:-4}
for r in $(seq 1 "$ROUNDS"); do
  for MODEL in protenix-v2 opendde; do
    for ARM in BASE C2FIX; do
      [ -s "perf/w6_c2fix/out/${ARM}_${MODEL}_298_r${r}.json" ] && continue
      $PY perf/w6_c2fix/arm.py --arm "$ARM" >/dev/null || exit 1
      $PY perf/w6_c2fix/fold_ab.py --arm "$ARM" --model "$MODEL" --size 298 --repeat 2 \
          --tag "_r${r}" >/dev/null || exit 1
      echo "round $r $MODEL $ARM done $(date -u +%H:%M:%S)"
    done
  done
done
echo "PAIRED DONE $(date -u +%H:%M:%S)"
