#!/bin/bash
# Waits for the bracket ladder, then re-runs the 512/544 pair WITH the lever counters.
#
# The 512 and 544 rungs that found the pothole predate the counter commit, so they cannot
# say whether the fused triangle attention served or declined. This is the decisive pair:
# same target, same chains, one bucket apart, both stamping TRIATT_FUSED_HIFI_STATS.
set -u
cd "$(dirname "$0")/../.."
card=${1:-0}
log=perf/bgx_size/out/counterpair_card$card.log
until grep -q "BRACKET DONE" perf/bgx_size/out/ladder_brk_card$card.log 2>/dev/null; do sleep 30; done
: > "$log"
for spec in "hIL2R 100 512" "hIL2R 146 544"; do
    set -- $spec
    echo "=== $(date -u +%FT%TZ) counter pair $1/$2 -> $3 tokens ===" >> "$log"
    timeout 5400 perf/bgx_size/run_rung.sh "$card" "fp$3_$1_$2" \
        --target "$1" --binder "$2" --rounds 3 --trajectories 1 --footprint >> "$log" 2>&1
    echo "=== rc=$? ===" >> "$log"
done
echo "=== COUNTER PAIR DONE $(date -u +%FT%TZ) ===" >> "$log"
