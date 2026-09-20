#!/bin/bash
# Re-record the p150a size-ladder for every model, now that the ladder reaches the board's
# 1536-token bar. One model at a time on card 0; each model writes its own fragment, so a
# pass that dies loses one model and not the campaign. Cheapest models first.
#
# PYTHONPATH is this worktree twice over: once for the gate's own imports (sys.path[0] is
# scripts/, not the repo root) and once through RELEASE_GATE_CENSUS_PYTHONPATH, which is the
# flag that exists so a pre-merge run censuses the checkout instead of an installed package.
set -u
WT=/home/ttuser/.coworker/wt/cov-ladder-below-bar-all-bhp150a
CARD=0
PY=/home/ttuser/kisoji_p2_fresh/env/bin/python3   # ttnn 0.68.0 + transformers 5.16.1: the stack pyproject declares, which the system python3 (ttnn 0.67.4) violates
cd "$WT" || exit 1
mkdir -p perf/sizegate/campaign/logs
for M in nesso1 protenix-v1 boltz2 rf3 openfold3 openbind protenix-v2 opendde esmfold2; do
  LOG=perf/sizegate/campaign/logs/$M.log
  if [ -f "perf/sizegate/campaign/logs/$M.done" ]; then continue; fi
  echo "=== $M start $(date -u +%FT%TZ) ===" >> "$LOG"
  PYTHONPATH="$WT" RELEASE_GATE_CENSUS_PYTHONPATH="$WT" \
  TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
  TT_BIO_LEASE_HOLDER=worker:cov-ladder-below-bar-all-bhp150a \
    "$PY" scripts/release_gate.py --model size-ladder --size-ladder-record \
      --size-ladder-fragment --size-ladder-models "$M" --load-ceiling 0 >> "$LOG" 2>&1
  RC=$?
  echo "=== $M rc=$RC $(date -u +%FT%TZ) ===" >> "$LOG"
  if [ $RC -eq 0 ]; then touch "perf/sizegate/campaign/logs/$M.done"; fi
done
touch perf/sizegate/campaign/logs/CAMPAIGN_FINISHED
