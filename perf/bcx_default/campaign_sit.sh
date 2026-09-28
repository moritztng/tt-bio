#!/bin/bash
# Two real PD-L1 campaigns on one qb2 card, same budget, same seed, same commit: the OLD
# default (one trajectory) and the NEW one (whatever the shipped default resolves to here).
# Sequential, because they share the card and a campaign is the thing being timed.
#   campaign_sit.sh [max_trajectories] [arms]     arms: space-separated tag:1|auto
set -uo pipefail
cd "$(dirname "$0")/../.."
budget=${1:-6}
arms=${2:-"old:1 auto:auto"}
o=perf/bcx_default/out; mkdir -p "$o"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -6 | cut -c1-160; } >> "$o/cotenants_campaign.txt";
    sleep 120; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for a in $arms; do
    IFS=: read -r tag want <<< "$a"
    echo "=== campaign $tag (want=$want, budget=$budget) $(date -u +%FT%TZ)"
    perf/bcx_default/campaign.sh "$tag" "$want" "$budget" > "$o/campaign_$tag.log" 2>&1 \
        || echo "  campaign $tag exited $?"
    echo "=== campaign $tag done $(date -u +%FT%TZ)"
done
echo "=== campaign sitting done $(date -u +%FT%TZ)"
