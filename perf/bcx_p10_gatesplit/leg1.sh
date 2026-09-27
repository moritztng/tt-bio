#!/bin/bash
# Leg 1 on pc card 0: the serial arm with the gate's hold split into phases, once per gate
# mode. Serial is one thread, so split 1 here only exercises the code path; the phase split of
# the split-0 arm is the reading.
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-4}; o=perf/bcx_p10_gatesplit/out
for a in p1:0 p2:1; do
    tag=${a%%:*}; split=${a##*:}
    echo "=== $tag serial split=$split $(date -u +%FT%TZ)"
    perf/bcx_p10_gatesplit/arm.sh "$tag" "$r" serial "$split" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== leg1 done $(date -u +%FT%TZ)"
