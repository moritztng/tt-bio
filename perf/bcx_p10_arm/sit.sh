#!/bin/bash
# The bcx-p10-arm sitting: harness-armed (a = old) against product-armed (b = new), `a b b a`.
#   sit.sh [rounds] [trajectories] [arms]    arms: space-separated tag:old|new list
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}; n=${2:-2}
arms=${3:-"a1:old b1:new b2:new a2:old"}
o=perf/bcx_p10_arm/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in $arms; do
    IFS=: read -r tag which <<< "$a"
    echo "=== $tag ($which, $r rounds, N=$n) $(date -u +%FT%TZ)"
    perf/bcx_p10_arm/arm.sh "$tag" "$which" "$r" "$n" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
