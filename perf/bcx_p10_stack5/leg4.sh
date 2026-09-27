#!/bin/bash
# duotraj leg 4 on the full stack: each trajectory in duo must equal the same trajectory serial.
# Two serial arms first give the determinism floor the comparison needs, then two duo arms.
#   leg4.sh [rounds]
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-4}; o=perf/bcx_p10_stack5/out
for a in e1:serial e2:duo e3:duo e4:serial; do
    tag=${a%%:*}; mode=${a##*:}
    echo "=== $tag ($mode, $r rounds) $(date -u +%FT%TZ)"
    perf/bcx_p10_stack5/arm.sh "$tag" "$r" "$mode" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== leg4 done $(date -u +%FT%TZ)"
