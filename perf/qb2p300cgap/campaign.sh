#!/usr/bin/env bash
# campaign.sh — the three p300c-uncovered legs, card 2, serial.
# Serial on one card on purpose: benchlock is host-scoped, so a concurrent draw on a sibling card
# would contaminate both readings through the shared host dispatch path. Each leg draws, then seeds
# its cell from a note built out of the draw index, then promotes the entry to the card layer, so a
# chain cut short still leaves whole legs behind rather than half a leg.
set -u
O=/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap/perf/qb2p300cgap
{
  echo "=== campaign start $(date -u +%FT%TZ) ==="
  for M in esmc-300m-single nesso1 rf3; do
    echo "--- $M draws $(date -u +%FT%TZ)"
    bash "$O/draw.sh" "$M" 5 2
    NOTE=$(python3 "$O/note.py" "$M") || { echo "$M: note failed, skipping reseed"; continue; }
    echo "--- $M reseed $(date -u +%FT%TZ)"
    bash "$O/reseed.sh" "$M" "$NOTE" 2
    echo "--- $M done $(date -u +%FT%TZ)"
  done
  echo "=== campaign done $(date -u +%FT%TZ) ==="
} >> "$O/campaign.out" 2>&1
