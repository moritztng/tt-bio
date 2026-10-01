#!/bin/bash
# The bcw-precision round sitting on qb2 card 3: TT_BIO_COTANGENT_B8 off against on, both on
# the user default (whatever `duotraj.auto_trajectories()` picks on this box), alternated at
# the process boundary so neither arm owns the quiet half of the sitting.
#   sit.sh [rounds] [arms]        arms: space-separated tag:1|auto[:VAR=value]
# The optional third field sets one env var for that arm only (the composed sitting's off arms).
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}
arms=${2:-"a1:auto:TT_BIO_COTANGENT_B8=0 b1:auto:TT_BIO_COTANGENT_B8=1 b2:auto:TT_BIO_COTANGENT_B8=1 a2:auto:TT_BIO_COTANGENT_B8=0 a3:auto:TT_BIO_COTANGENT_B8=0 b3:auto:TT_BIO_COTANGENT_B8=1"}
o=perf/bcw_precision/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in $arms; do
    IFS=: read -r tag want flag <<< "$a"
    echo "=== $tag (want=$want${flag:+, $flag}, $r rounds) $(date -u +%FT%TZ)"
    env ${flag:-} perf/bcw_precision/arm.sh "$tag" "$r" "$want" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
