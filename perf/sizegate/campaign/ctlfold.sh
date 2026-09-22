#!/bin/bash
# One boltz2 rung at 256 recorded to scratch: the live control for the new per-rung clock,
# per-rep array and structure read, and the direct read of what SDPA_WIDE_K resolves to on
# the engine this branch ships. Scratch baseline, so no committed fragment is touched.
set -u
WT=/home/ttuser/.coworker/wt/cov-ladder-p150a-p3
PY=/home/ttuser/kisoji_p2_fresh/env/bin/python3
cd "$WT" || exit 1
OUT=$WT/perf/sizegate/campaign/scratch_b2_256.json
PYTHONPATH="$WT" RELEASE_GATE_CENSUS_PYTHONPATH="$WT" \
RELEASE_GATE_SIZE_WORKDIR="$WT/perf/sizegate/work-card0" \
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
TT_BIO_LEASE_HOLDER=worker:cov-ladder-p150a-p3 \
  "$PY" scripts/release_gate.py --model size-ladder --size-ladder-record \
    --size-ladder-baseline "$OUT" --size-ladder-models boltz2 \
    --size-ladder-rungs 256 --load-ceiling 0
echo "=== rc=$? $(date -u +%FT%TZ) ==="
