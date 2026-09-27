#!/bin/bash
# The milestone sitting on one qb2 card: footprint, then two alternations of serial and duo at
# the process boundary (`s d d s`, then `d s s d`), then the accuracy legs. Every arm has mean
# position 2.5 inside its half. A co-tenant snapshot every 60 s goes to out/cotenants.txt.
#   sit.sh [rounds] [phases]      phases: any of fp,time,acc (default all three)
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}; phases=${2:-fp,time,acc}
o=perf/bcx_p10_stack5/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
case $phases in *fp*)
    echo "=== fp $(date -u +%FT%TZ)"
    perf/bcx_p10_stack5/footprint.sh fp 3 > "$o/fp.log" 2>&1 || echo "  fp exited $?" ;; esac
case $phases in *time*)
    for a in s1:serial d1:duo d2:duo s2:serial d3:duo s3:serial s4:serial d4:duo; do
        tag=${a%%:*}; mode=${a##*:}
        echo "=== $tag ($mode, $r rounds) $(date -u +%FT%TZ)"
        perf/bcx_p10_stack5/arm.sh "$tag" "$r" "$mode" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
    done ;; esac
case $phases in *acc*)
    echo "=== acc $(date -u +%FT%TZ)"
    perf/bcx_p10_stack5/acc.sh > "$o/acc.log" 2>&1 || echo "  acc exited $?" ;; esac
echo "=== sitting done $(date -u +%FT%TZ)"
