#!/usr/bin/env bash
# of3t-modelever: the device arms in one session on qb2 card 1, A/A first, banked arms skipped.
set -uo pipefail
cd /home/ttuser/.coworker/wt/of3t-modelever
L=/home/ttuser/of3t_modelever
for spec in "$@"; do
  TAG=${spec%%:*}; MODE=${spec##*:}
  if [ -s "perf/of3t_modelever/DEV_${TAG}.json" ] && [ -s "$L/dev_${TAG}.pt" ]; then
    echo "=== $TAG already banked, skipping ==="; continue
  fi
  bash perf/of3t_modelever/arm.sh "$TAG" "$MODE" > "$L/dev_${TAG}.log" 2>&1
  echo "=== $TAG rc $? $(date -u +%FT%TZ) ==="
done
echo "=== CHAIN DONE $(date -u +%FT%TZ) ==="
