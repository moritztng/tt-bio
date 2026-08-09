#!/bin/bash
# release_gate.py for one arm, the four legs W11 gated. Resumable: a leg whose log already
# ends in a verdict line is skipped.
#   bash perf/w6_c2fix/gate.sh <BASE|C2FIX>
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-c2fix-land
export RELEASE_GATE_MSA_DIR="$HOME/w6_gate_msa"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export OPENDDE_DOCKQ_PYTHON=$HOME/w6_dockq_py
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
ARM=$1
mkdir -p perf/w6_c2fix/out
$PY perf/w6_c2fix/arm.py --arm "$ARM" >/dev/null || exit 1
for LEG in protenix-v2 opendde opendde-abag capacity; do
  LOG="perf/w6_c2fix/out/rg_${ARM}_${LEG}.log"
  if [ -s "$LOG" ] && grep -qE "^(GATE|RESULT|OVERALL)" "$LOG"; then echo "SKIP $LEG"; continue; fi
  echo "=== rg $ARM $LEG start $(date -u +%H:%M:%S) ==="
  $PY scripts/release_gate.py --model "$LEG" > "$LOG" 2>&1
  echo "=== rg $ARM $LEG rc=$? $(date -u +%H:%M:%S) ==="
  tail -6 "$LOG"
done
echo "GATE SWEEP DONE $ARM $(date -u +%H:%M:%S)"
