#!/usr/bin/env bash
# Wait for an explicit pid to exit, then run a rung on a card.
# Usage: chain.sh <pid-to-wait-for> <tokens> <card>
#
# `kill -0` on an explicit pid, not a pgrep pattern: a pattern match would find this
# wrapper's own argv.
set -u
PID=$1; N=$2; CARD=$3
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
exec bash perf/of3t_crop768/run_rung_nolock.sh "$N" "$CARD"
