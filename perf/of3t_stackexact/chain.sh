#!/usr/bin/env bash
# of3t-stackexact: the ladder in one session on qb2 card 1, A/A first, banked rungs skipped.
#   chain.sh SHIP_A:none S:softmax L:layer_norm SHIP_B:none SL:softmax,layer_norm
set -uo pipefail
cd /home/ttuser/.coworker/wt/of3t-stackexact
L=/home/ttuser/of3t_stackexact
mkdir -p "$L"
for spec in "$@"; do
  TAG=${spec%%:*}; SC=${spec#*:}
  if [ -s "perf/of3t_stackexact/DEV_${TAG}.json" ] && [ -s "$L/dev_${TAG}.pt" ]; then
    echo "=== $TAG already banked, skipping ==="; continue
  fi
  bash perf/of3t_stackexact/arm.sh "$TAG" "$SC" > "$L/dev_${TAG}.log" 2>&1
  echo "=== $TAG rc $? $(date -u +%FT%TZ) ==="
done
echo "=== CHAIN DONE $(date -u +%FT%TZ) ==="
