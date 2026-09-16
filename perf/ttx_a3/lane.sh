#!/usr/bin/env bash
# One serial size-ladder lane: wait for whatever ladder arm is running to finish, then splice +
# check each model in turn. Two ladder arms on one box corrupt each other -- they share
# perf/sizegate/work and clean it at exit -- so this waits rather than fanning out.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2
cd "$WT" || exit 1
waited=0
while pgrep -f 'release_gate.py --model size-ladder' > /dev/null; do
  [ "$waited" -ge 5400 ] && { echo "lane: gave up waiting after 90 min"; exit 1; }
  sleep 60; waited=$((waited + 60))
done
exec bash perf/ttx_a3/ladder_campaign.sh "$@"
