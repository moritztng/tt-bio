#!/bin/bash
# One sitting, three arms, six processes: off cmp on on cmp off.
#
# The order is a palindrome, so every arm's mean process position is 2.5 and a monotone drift
# over the sitting cancels for all three equally rather than loading one. Arms alternate at the
# PROCESS boundary, never inside one, because the levers are resolved at import.
#
# 9 rounds a process, round 1 of each dropped as its compile round: 8 timed rounds a process,
# 16 an arm. Nine is the ceiling, see arm.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
r=${ROUNDS:-9}
for a in p1_off:off p2_cmp:cmp p3_on:on p4_on:on p5_cmp:cmp p6_off:off; do
    tag=${a%%:*}; mode=${a##*:}
    echo "=== $(date -u +%H:%M:%SZ) $tag ($mode) load1=$(cut -d' ' -f1 /proc/loadavg) ==="
    perf/bcx_p10_stack3/arm.sh "$tag" "$r" "$mode"
done
echo "=== $(date -u +%H:%M:%SZ) sitting complete ==="
