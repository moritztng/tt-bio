#!/usr/bin/env bash
# of3t-angle: the four device arms, in one session on qb2 card 1, interleaved A/A first.
# SHIP_A and SHIP_B are two runs of the same lever: the A/A floor, taken before the levers and
# read as the size of any arm-to-arm difference that is not a lever.
set -uo pipefail
cd /home/ttuser/.coworker/wt/of3t-angle
L=/home/ttuser/of3t_angle
for spec in SHIP_A:all SHIP_B:all VERB:ceiling_hf MODW:ceiling_hf3; do
  TAG=${spec%%:*}; LEV=${spec##*:}
  if [ -s "perf/of3t_angle/DEV_${TAG}.json" ] && [ -s "$L/dev_${TAG}.pt" ]; then
    echo "=== $TAG already banked, skipping ==="
    continue
  fi
  bash perf/of3t_angle/devarms.sh "$TAG" "$LEV" > "$L/dev_${TAG}.log" 2>&1
  echo "=== $TAG rc $? $(date -u +%FT%TZ) ==="
done
echo "=== CHAIN DONE $(date -u +%FT%TZ) ==="
