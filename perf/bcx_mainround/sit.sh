#!/bin/bash
# bcx-mainround: ff538cafd (arm A, ./.armA at 10e626bd0 = ff538cafd + the headline harness) against
# origin/main (arm B, this tree), alternating A B A B on one qb2 card, each through the headline's
# arm.sh (N=3, 9 rounds, every lever env var unset).
#   sit.sh [rounds] [trajectories] [tags]
set -uo pipefail
B=$(cd "$(dirname "$0")/../.." && pwd); A=$B/.armA
r=${1:-9}; n=${2:-3}; tags=${3:-"a1 b1 a2 b2"}
o=$B/perf/bcx_mainround/out; mkdir -p "$o"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-mainround
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for tag in $tags; do
    case $tag in a*) t=$A;; *) t=$B;; esac
    echo "=== $tag $(git -C "$t" rev-parse --short HEAD) $(date -u +%FT%TZ)"
    "$t/perf/bcx_p10_headline/arm.sh" "$tag" "$r" "$n" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
    rm -rf "${o:?}/${tag:?}"; cp -r "$t/perf/bcx_p10_headline/out/$tag" "$o/$tag"
done
echo "=== sitting done $(date -u +%FT%TZ)"
