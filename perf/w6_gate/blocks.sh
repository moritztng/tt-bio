#!/bin/bash
# Re-baseline W6's Pairformer block numbers on qb1 card 0 at ttnn 0.67.4, per arm and per size.
# W6 measured on qb2 at 0.68.0, and WARROOM section 3 says a qb2 number is a ratio, not a result.
#
# N=298 is in here on purpose. W6 benchmarked N=320, the padded length, and the fold passes the
# logical 298. That difference is the whole C1 story, so the block bench has to be run at both.
#
#   bash perf/w6_gate/blocks.sh
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-fold-parity-gate || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-fold-parity-gate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
OUT=perf/w6_gate/out
mkdir -p "$OUT/logs"

# Roofs first, on BASE: W6's percentages are against qb2 card 1 (377 GB/s copy, 102.4 TFLOP/s)
# and this card's differ. Measure, never inherit.
$PY perf/w6_gate/arm.py --arm BASE >/dev/null
if [ ! -s "$OUT/roofs_qb1c0.json" ]; then
  echo "=== roofs BASE $(date -u +%H:%M:%S) ==="
  $PY perf/attn_block/op_split.py --mode roofs --out "$OUT/roofs_qb1c0.json" \
      > "$OUT/logs/roofs_qb1c0.log" 2>&1
  echo "   rc=$? $(date -u +%H:%M:%S)"
fi

# C2 and ALL are omitted: they raise AttributeError in _configure_active_compute_grid at device
# open on any grid that is not the module default (11x10), which is every p150a (13x10).
for ARM in BASE C1 C2FIX C3 C4 ALLFIX; do
  $PY perf/w6_gate/arm.py --arm "$ARM" >/dev/null || { echo "ARM FAIL $ARM"; exit 1; }
  for N in 320 298 128; do
    J="$OUT/mods_n${N}_qb1c0_${ARM}.json"
    [ -s "$J" ] && { echo "SKIP $ARM n=$N"; continue; }
    echo "=== block $ARM n=$N $(date -u +%H:%M:%S) ==="
    $PY perf/attn_block/op_split.py --mode mods --n "$N" --iters 5 --out "$J" \
        > "$OUT/logs/mods_n${N}_${ARM}.log" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then echo "   rc=$RC"; tail -4 "$OUT/logs/mods_n${N}_${ARM}.log"; fi
    grep -E "^FULL_BLOCK" "$OUT/logs/mods_n${N}_${ARM}.log" 2>/dev/null
  done
done
$PY perf/w6_gate/arm.py --arm BASE >/dev/null
echo "BLOCKS DONE, worktree restored to BASE"
