#!/usr/bin/env bash
# Wait for an explicit pid, then run a ladder of rungs on one card, in order.
# Usage: chain2.sh <pid-to-wait-for> <card> <tokens:arm> [<tokens:arm> ...]
#
# `kill -0` on an explicit pid, not a pgrep pattern: a pattern would match this wrapper's argv.
set -u
PID=$1; CARD=$2; shift 2
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
for spec in "$@"; do
  N=${spec%%:*}; ARM=${spec##*:}
  echo "=== rung $N arm $ARM $(date -u +%FT%TZ) ==="
  bash perf/of3t_crop768/run_rung2.sh "$N" "$CARD" "$ARM" || true
done
