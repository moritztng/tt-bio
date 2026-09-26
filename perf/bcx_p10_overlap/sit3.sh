#!/bin/bash
# Leg 3's mechanism sitting: four processes alternating at the PROCESS boundary, off, on, on,
# off, so a drift over the sitting cancels rather than loading one arm. The only difference
# between the arms is the --hostload worker; both carry all three wave-10 levers and the
# timeline. 9 rounds a process, round 1 dropped as the compile.
set -euo pipefail
cd "$(dirname "$0")/../.."
r=${ROUNDS:-9}
for a in q1_off:0 q2_on:1 q3_on:1 q4_off:0; do
    tag=${a%%:*}; on=${a##*:}
    echo "=== $(date -u +%H:%M:%SZ) $tag (hostload=$on) ==="
    HOSTLOAD=$on perf/bcx_p10_overlap/arm.sh "$tag" "$r" off
done
echo "=== $(date -u +%H:%M:%SZ) sitting complete ==="
