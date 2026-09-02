#!/usr/bin/env bash
# campaign.sh — the three p300c-uncovered legs on card 2, serial and idempotent.
#
# Serial on one card on purpose: benchlock is host-scoped, so a concurrent draw on a sibling card
# would contaminate both readings through the shared host dispatch path.
#
# Idempotent per leg. A leg is skipped when it already has a cells.p300c.models entry AND at least
# MIN_DRAWS drawn values in the index, so a chain that is cut short and relaunched resumes at the
# first unfinished leg instead of redrawing what is already seeded. Each leg draws, then seeds its
# own cell from a note built out of the draw index, then promotes the entry to the card layer, so a
# chain cut short leaves whole legs behind rather than half a leg.
#
# No wait-for-quiet loop here any more. draw.sh resets the board before every draw and benchlock
# still refuses to measure under load once it holds the lock, and the previous version's own
# 8 h quiet wait is what left this task with nothing measured for three hours: it sat outside the
# lock waiting for a box that a peer's benchlocked ladder chain was legitimately using, which is
# not noise to wait out, it is the lock working. Queue on the lock instead and let benchlock's
# BENCHLOCK_LOAD_WAIT_S handle actual co-tenant noise.
set -u
O=/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap/perf/qb2p300cgap
WT=/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap
export BENCHLOCK_LOAD_WAIT_S=2400        # once we hold the lock, still refuse to measure loud
export BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-21600}   # a peer ladder chain can hold it for hours
MIN_DRAWS=${MIN_DRAWS:-5}
{
  echo "=== campaign start $(date -u +%FT%TZ) ==="
  for M in esmc-300m-single nesso1 rf3; do
    if python3 "$O/leg_done.py" "$M" "$MIN_DRAWS"; then
      echo "--- $M already seeded with >= $MIN_DRAWS draws, skipping"
      continue
    fi
    echo "--- $M draws $(date -u +%FT%TZ)"
    bash "$O/draw.sh" "$M" "$MIN_DRAWS" 2
    NOTE=$(python3 "$O/note.py" "$M") || { echo "$M: note failed, skipping reseed"; continue; }
    echo "--- $M reseed $(date -u +%FT%TZ)"
    bash "$O/reseed.sh" "$M" "$NOTE" 2
    echo "--- $M done $(date -u +%FT%TZ)"
  done
  echo "=== campaign done $(date -u +%FT%TZ) ==="
} >> "$O/campaign.out" 2>&1
