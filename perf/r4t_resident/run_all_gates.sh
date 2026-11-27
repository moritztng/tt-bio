#!/usr/bin/env bash
set -euo pipefail

# Run all r4t-resident hardware verification gates sequentially on one pinned Blackhole chip.
# Usage:
#   TT_VISIBLE_DEVICES=24 TT_BIO_LEASE_CARDS=24 TT_BIO_LEASE_HOLDER=worker:r4t-resident \
#     bash perf/r4t_resident/run_all_gates.sh <python_bin> <batches_dir> <out_dir>

PY="${1:?python binary required}"
BDIR="${2:?batches directory required}"
OUT="${3:?output directory required}"
mkdir -p "$OUT"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TT_BIO_SHA="$(git -C "$REPO" rev-parse --short=9 HEAD)"

echo "=== r4t-resident gates: sha=$TT_BIO_SHA card=${TT_VISIBLE_DEVICES:-unset} out=$OUT ==="

echo "--- [Gate 1/3] Hardware unit suites (u32 smoke, DeviceHeads + broken arms, DeviceFolds, DeviceOptimizer) ---"
timeout 900 "$PY" "$REPO/perf/r4t_resident/test_resident_units.py" \
    --batch8 "$BDIR/batch08.npz" \
    --out "$OUT/units.json" 2>&1 | tee "$OUT/units.log"

echo "--- [Gate 2/3] 20-step OPT-PARITY (<=1e-4 vs float64), D37 trajectory, and BROKEN-ARM (ns=4, beta2=0.99) ---"
timeout 1200 "$PY" "$REPO/perf/r4t_resident/grade_opt_parity.py" \
    --batches "$BDIR/batch*.npz" \
    --steps 20 \
    --ns-mode split2 \
    --out "$OUT/opt_parity.json" 2>&1 | tee "$OUT/opt_parity.log"

echo "--- [Gate 3/3] End-to-end --resident timing (18 steps) + HOST-BYTES (--pcie-meter sites) ---"
BATCHES="$(ls "$BDIR"/batch0[0-8].npz | paste -sd, -)"
timeout 900 "$PY" "$REPO/perf/r4t_step/trunk_step.py" \
    --resident --optimizer muon --cand-levers serve --cand-pair card --win-pair card \
    --batch "$BATCHES" --steps 18 --pcie-meter sites --out "$OUT/e2e_run" \
    2>&1 | tee "$OUT/e2e_resident.jsonl"

echo "=== ALL 3 GATES COMPLETE ==="
