#!/bin/bash
# Start an arm on the first qb2 card that is GENUINELY idle, and nothing sooner.
#
# qb2 carries bcw-land's gate and bcp-evo, and both run short sittings back to back: at 17:14Z
# cards 1 and 3 had held a fd for 51 and 61 seconds respectively, which is a sweep between
# launches, not a free card. A snapshot of `fuser` cannot tell those apart from an idle chip, so
# this waits for a card to be continuously empty for IDLE_FOR seconds before taking it. A row
# whose worker is mid-sweep never shows a gap that long; a card nobody wants shows one immediately.
#
# It launches ONE arm and exits. It never touches card 2, which arm B is holding, and it gives up
# at DEADLINE rather than sitting on the box forever.
set -uo pipefail
cd /home/ttuser/.coworker/wt/bcw-accept
IDLE_FOR=${IDLE_FOR:-1200}
DEADLINE=$(( $(date +%s) + ${LIVE_FOR:-86400} ))
tag=${1:-armA1}; arm=${2:-a}; seed=${3:-100}; cards=${CARDS:-"0 1 3"}
log=perf/bcw_accept/out/wait_card.log
mkdir -p perf/bcw_accept/out
declare -A since
echo "=== $(date -u +%FT%TZ) waiting for a card idle $IDLE_FOR s, for arm $arm tag=$tag seed=$seed cards=$cards, pid $$" >> "$log"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    now=$(date +%s)
    for n in $cards; do
        if fuser "/dev/tenstorrent/$n" >/dev/null 2>&1; then
            unset "since[$n]"
            continue
        fi
        [ -n "${since[$n]:-}" ] || since[$n]=$now
        held=$(( now - ${since[$n]} ))
        if [ "$held" -ge "$IDLE_FOR" ]; then
            echo "=== $(date -u +%FT%TZ) card $n empty for ${held}s, taking it for arm $arm tag=$tag" >> "$log"
            # Re-check immediately before the launch: the gap can close in the last second.
            if fuser "/dev/tenstorrent/$n" >/dev/null 2>&1; then
                echo "    card $n taken by someone in the last tick, backing off" >> "$log"
                unset "since[$n]"
                continue
            fi
            exec bash perf/bcw_accept/campaign.sh "$arm" "$n" "$tag" "$seed"
        fi
    done
    sleep 60
done
echo "=== $(date -u +%FT%TZ) deadline reached, no card was idle $IDLE_FOR s, giving up" >> "$log"
