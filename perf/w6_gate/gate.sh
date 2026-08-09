#!/bin/bash
# release_gate.py across the four legs that matter to this gate, for one arm.
# Resumable: a leg whose log already ends in a verdict line is skipped.
#   bash perf/w6_gate/gate.sh <ARM>
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-fold-parity-gate || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-fold-parity-gate
export RELEASE_GATE_MSA_DIR="$HOME/w6_gate_msa"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
export OPENDDE_DOCKQ_PYTHON=$HOME/w6_dockq_py

ARM=$1
mkdir -p perf/w6_gate/out
$PY perf/w6_gate/arm.py --arm "$ARM" >/dev/null || exit 1

for LEG in protenix-v2 opendde opendde-abag capacity; do
  LOG="perf/w6_gate/out/rg_${ARM}_${LEG}.log"
  if [ -s "$LOG" ] && grep -qE "^(GATE|RESULT|OVERALL)" "$LOG"; then echo "SKIP $LEG"; continue; fi
  echo "=== rg $ARM $LEG start $(date -u +%H:%M:%S) ==="
  $PY scripts/release_gate.py --model "$LEG" > "$LOG" 2>&1
  echo "=== rg $ARM $LEG rc=$? $(date -u +%H:%M:%S) ==="
  tail -6 "$LOG"
done
$PY perf/w6_gate/arm.py --arm BASE >/dev/null
