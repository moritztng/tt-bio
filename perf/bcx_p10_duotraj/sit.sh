#!/bin/bash
# One sitting: four processes alternating serial / duo / duo / serial at the process boundary,
# so a drift in the box's load over the sitting cancels rather than loading one arm. Every arm
# has mean position 2.5.
#
#   sit.sh [rounds]
#
# Rounds are low on purpose. pc has 31 GB of host memory against co-tenants holding 14-20, and
# two live trajectories plateau around 11 GB resident -- so the arm that has to fit is the duo
# arm, and both arms run the same length or the comparison is of two different things.
set -euo pipefail
cd "$(dirname "$0")/../.."
r=${1:-6}
for a in s1:serial d1:duo d2:duo s2:serial; do
    tag=${a%%:*}; mode=${a##*:}
    echo "=== $tag ($mode, $r rounds) $(date -u +%FT%TZ) ==="
    perf/bcx_p10_duotraj/arm.sh "$tag" "$r" "$mode" \
        > "perf/bcx_p10_duotraj/out/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ) ==="
