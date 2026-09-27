#!/bin/bash
# The tritraj sitting on one qb2 card: N=2 vs N=3 interleaved, alternated at the process
# boundary (`2 3 3 2`, then `3 2 2 3`), then one serial N=3 arm as the leg-4 reference.
# A co-tenant snapshot every 60 s goes to out/cotenants.txt.
#   sit.sh [rounds] [arms] [prefix]    arms: space-separated tag:N:mode list
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}
arms=${2:-"a1:2:inter b1:3:inter b2:3:inter a2:2:inter b3:3:inter a3:2:inter a4:2:inter b4:3:inter e1:3:serial"}
o=perf/bcx_p10_tritraj/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in $arms; do
    IFS=: read -r tag n mode <<< "$a"
    echo "=== $tag (N=$n $mode, $r rounds) $(date -u +%FT%TZ)"
    perf/bcx_p10_tritraj/arm.sh "$tag" "$r" "$n" "$mode" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
