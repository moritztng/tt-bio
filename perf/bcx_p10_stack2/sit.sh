#!/bin/bash
# One sitting: arms alternating at the PROCESS boundary in the order off, on, on, off, so a
# drift over the sitting cancels rather than loading one arm. 9 rounds a process, round 1 of
# each dropped as its compile round.
set -euo pipefail
cd "$(dirname "$0")/../.."
r=${ROUNDS:-9}
for a in p1_off:off p2_on:on p3_on:on p4_off:off; do
    tag=${a%%:*}; mode=${a##*:}
    echo "=== $(date -u +%H:%M:%SZ) $tag ($mode) ==="
    perf/bcx_p10_stack2/arm.sh "$tag" "$r" "$mode"
done
echo "=== $(date -u +%H:%M:%SZ) sitting complete ==="
