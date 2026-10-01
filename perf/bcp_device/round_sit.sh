#!/bin/bash
# A lever's round sitting: arms alternated at the process boundary, ABBAAB, on one card.
#   round_sit.sh <FLAG> [rounds]     e.g. round_sit.sh TT_BIO_LNBW_FUSED 9
set -uo pipefail
cd "$(dirname "$0")/../.."
flag=$1; r=${2:-9}
o=perf/bcp_device/out/round; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in off1:0 on1:1 on2:1 off2:0 off3:0 on3:1; do
    IFS=: read -r tag v <<< "$a"
    echo "=== $tag ($flag=$v, $r rounds) $(date -u +%FT%TZ)"
    perf/bcp_device/round_arm.sh "$tag" "$r" "$flag=$v" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
