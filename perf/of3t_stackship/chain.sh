#!/usr/bin/env bash
# of3t-stackship: the off switch first, then the shipped arm, one session on qb2 card 1.
set -uo pipefail
cd /home/ttuser/.coworker/wt/of3t-stackship
L=/home/ttuser/of3t_stackship
for spec in "$@"; do
  TAG=${spec%%:*}; MODE=${spec#*:}
  bash perf/of3t_stackship/arm.sh "$TAG" "$MODE" > "$L/dev_${TAG}.log" 2>&1
  echo "=== $TAG rc $? $(date -u +%FT%TZ) ==="
done
echo "=== CHAIN DONE $(date -u +%FT%TZ) ==="
