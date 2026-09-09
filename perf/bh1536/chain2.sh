#!/bin/bash
# Runs the next batch once chain.sh is gone, so the two never contend for the card. Waits on the
# ABSENCE of a chain.sh process rather than on a pid handed in at launch: the pid that was handed
# in the first time was the launching shell, not the chain, and that shell can exit first.
# chain.sh skips any rung already in results.jsonl, so re-running this after a relaunch is free.
WT=/home/ttuser/.coworker/wt/bh-1536-structure
cd $WT || exit 1
while pgrep -f "bh1536/chain\.sh" > /dev/null; do sleep 30; done
echo "chain.sh gone at $(date -u +%FT%TZ), starting"
exec ./perf/bh1536/chain.sh "$@"
