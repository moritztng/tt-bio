#!/bin/bash
# Legs 3 and 4 on one qb2 card. Duo with the old gate (o) against duo with the split gate (n),
# interleaved at the process boundary `o n n o | n o o n` so every arm has mean position 2.5
# inside its half, between two serial arms (one per gate mode) that give the equality check
# its reference and its determinism floor. Co-tenant snapshot every 60 s to out/cotenants.txt.
#   TT_VISIBLE_DEVICES=<card> sit.sh [rounds]
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}; o=perf/bcx_p10_gatesplit/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in s1:serial:0 o1:duo:0 n1:duo:1 n2:duo:1 o2:duo:0 n3:duo:1 o3:duo:0 o4:duo:0 n4:duo:1 s2:serial:1; do
    IFS=: read -r tag mode split <<< "$a"
    echo "=== $tag ($mode split=$split, $r rounds) $(date -u +%FT%TZ)"
    perf/bcx_p10_gatesplit/arm.sh "$tag" "$r" "$mode" "$split" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
