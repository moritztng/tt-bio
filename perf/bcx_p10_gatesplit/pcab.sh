#!/bin/bash
# Duo old gate (o) against duo split gate (n) on pc card 0, interleaved `o n n o | n o o n`.
# Three rounds a trajectory is what pc's RAM holds in duo; round 1 is dropped, so each arm
# gives ~4 warm rounds pro rata. Speed only: pc card 0's matmul fault bars an equality check.
#   pcab.sh [rounds]
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-3}; o=perf/bcx_p10_gatesplit/out
for a in q1:0 q2:1 q3:1 q4:0 q5:1 q6:0 q7:0 q8:1; do
    tag=${a%%:*}; split=${a##*:}
    echo "=== $tag duo split=$split $(date -u +%FT%TZ) mem_avail $(awk '/MemAvailable/{print $2}' /proc/meminfo)"
    perf/bcx_p10_gatesplit/arm.sh "$tag" "$r" duo "$split" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== pcab done $(date -u +%FT%TZ)"
