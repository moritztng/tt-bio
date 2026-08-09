#!/bin/bash
# W6 gate fold sweep driver. One process per (arm, model, size): arm.py materialises the arm,
# fold_ab.py folds it. fold_ab.py skips a run whose JSON already exists, so re-running this
# script resumes rather than repeating device time.
#
#   bash perf/w6_gate/sweep.sh <size> <model> <repeat> <arm> [arm...]
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-fold-parity-gate || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-fold-parity-gate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3

SIZE=$1; MODEL=$2; REP=$3; shift 3
mkdir -p perf/w6_gate/out/logs

for ARM in "$@"; do
  TAG="${ARM}_${MODEL}_${SIZE}"
  if [ -f "perf/w6_gate/out/${TAG}.json" ]; then echo "SKIP $TAG (done)"; continue; fi
  echo "=== $TAG start $(date -u +%H:%M:%S) ==="
  $PY perf/w6_gate/arm.py --arm "$ARM" >/dev/null || { echo "ARM FAIL $ARM"; exit 1; }
  $PY perf/w6_gate/fold_ab.py --arm "$ARM" --model "$MODEL" --size "$SIZE" --repeat "$REP" \
      > "perf/w6_gate/out/logs/${TAG}.log" 2>&1
  RC=$?
  echo "=== $TAG rc=$RC $(date -u +%H:%M:%S) ==="
  if [ $RC -ne 0 ]; then tail -25 "perf/w6_gate/out/logs/${TAG}.log"; fi
done
$PY perf/w6_gate/arm.py --arm BASE >/dev/null
echo "ALL DONE, worktree restored to BASE"
