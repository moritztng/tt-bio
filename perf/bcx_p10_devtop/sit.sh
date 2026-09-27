#!/bin/bash
# The bcx-p10-devtop sitting on qb2 card 0: N=3 stack (a) vs N=3 stack + TT_BIO_QKV_GRAD_JOIN
# (b), alternated at the process boundary `a b b a b a a b`. Co-tenant snapshot every 60 s.
#   sit.sh [rounds] [arms]    arms: space-separated tag:join list
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}
arms=${2:-"a1:0 b1:1 b2:1 a2:0 b3:1 a3:0 a4:0 b4:1"}
o=perf/bcx_p10_devtop/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in $arms; do
    IFS=: read -r tag join <<< "$a"
    echo "=== $tag (join=$join, $r rounds) $(date -u +%FT%TZ)"
    perf/bcx_p10_devtop/arm.sh "$tag" "$r" "$join" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
