#!/bin/bash
# A trace sitting on one qb2 card: arms alternated at the process boundary, with a co-tenant
# snapshot every 60 s in out/cotenants.txt.
#   sit.sh <rounds> "<tag:N:mode:trace> ..."
set -uo pipefail
cd "$(dirname "$0")/../.."
r=$1; arms=$2
o=perf/bcx_p10_trace/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in $arms; do
    IFS=: read -r tag n mode trace <<< "$a"
    echo "=== $tag (N=$n $mode trace=$trace, $r rounds) $(date -u +%FT%TZ)"
    perf/bcx_p10_trace/arm.sh "$tag" "$r" "$n" "$mode" "$trace" > "$o/$tag.log" 2>&1 \
        || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
