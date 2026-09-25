#!/usr/bin/env bash
# of3t-stackship: training step cost, arms interleaved on qb2 card 1.
#   stepchain.sh TAG:on|off ...
set -uo pipefail
cd /home/ttuser/.coworker/wt/of3t-stackship
L=/home/ttuser/of3t_stackship
for spec in "$@"; do
  TAG=${spec%%:*}; X=${spec#*:}
  bash perf/of3t_stackship/stepcost.sh "$TAG" "$X" > "$L/step_${TAG}.log" 2>&1
  echo "=== $TAG rc $? $(date -u +%FT%TZ) ==="
done
echo "=== STEPCHAIN DONE $(date -u +%FT%TZ) ==="
