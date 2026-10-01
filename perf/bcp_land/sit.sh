#!/bin/bash
# The bcx-default round sitting on one qb2 card: the OLD default (one trajectory, no gate)
# against the NEW one (whatever `duotraj.auto_trajectories()` picks on this box), alternated at
# the process boundary so neither arm owns the quiet half of the sitting.
#   sit.sh [rounds] [arms]        arms: space-separated tag:1|auto
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}
arms=${2:-"a1:1 b1:auto b2:auto a2:1 b3:auto a3:1 a4:1 b4:auto"}
o=perf/bcp_land/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in $arms; do
    IFS=: read -r tag want <<< "$a"
    echo "=== $tag (want=$want, $r rounds) $(date -u +%FT%TZ)"
    perf/bcp_land/arm.sh "$tag" "$r" "$want" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
