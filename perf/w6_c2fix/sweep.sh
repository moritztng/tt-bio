#!/bin/bash
# BASE vs C2FIX, both models, both sizes, on card 1. Resumable: fold_ab.py skips an existing JSON.
# Arms are interleaved per (model,size) so the two folds of a pair see the same box contention.
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-c2fix-land
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
for pair in "protenix-v2 298 5" "opendde 298 5" "protenix-v2 117 3" "opendde 117 3"; do
  set -- $pair; MODEL=$1; SIZE=$2; REP=$3
  for ARM in BASE C2FIX; do
    if [ -s "perf/w6_c2fix/out/${ARM}_${MODEL}_${SIZE}.json" ]; then echo "SKIP $ARM $MODEL $SIZE"; continue; fi
    echo "=== $ARM $MODEL $SIZE start $(date -u +%H:%M:%S) ==="
    $PY perf/w6_c2fix/arm.py --arm "$ARM" || exit 1
    $PY perf/w6_c2fix/fold_ab.py --arm "$ARM" --model "$MODEL" --size "$SIZE" --repeat "$REP"
    echo "=== $ARM $MODEL $SIZE rc=$? $(date -u +%H:%M:%S) ==="
  done
done
$PY perf/w6_c2fix/arm.py --arm C2FIX
echo "SWEEP DONE $(date -u +%H:%M:%S)"
