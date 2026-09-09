#!/bin/bash
# Third batch. Waits for BOTH earlier chains: chain2.sh execs chain.sh, and there is a poll-length
# window where chain1 has exited and chain2 has not exec'd yet, so waiting on chain.sh alone would
# start inside it and contend for the card.
WT=/home/ttuser/.coworker/wt/bh-1536-structure
cd $WT || exit 1
while pgrep -f "bh1536/chain\.sh|bh1536/chain2\.sh" > /dev/null; do sleep 30; done
echo "earlier chains gone at $(date -u +%FT%TZ), starting"
exec ./perf/bh1536/chain.sh "$@"
