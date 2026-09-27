#!/bin/bash
# The bcx-p10-headline sitting: the same shipped tree four times (A/A), so the spread is the floor.
#   sit.sh [rounds] [trajectories] [tags]
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}; n=${2:-3}; tags=${3:-"h1 h2 h3 h4"}
o=perf/bcx_p10_headline/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for tag in $tags; do
    echo "=== $tag ($r rounds, N=$n) $(date -u +%FT%TZ)"
    perf/bcx_p10_headline/arm.sh "$tag" "$r" "$n" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
