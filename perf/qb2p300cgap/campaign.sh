#!/usr/bin/env bash
# campaign.sh — the three p300c-uncovered legs, card 2, serial.
#
# Serial on one card on purpose: benchlock is host-scoped, so a concurrent draw on a sibling card
# would contaminate both readings through the shared host dispatch path. Each leg draws, then seeds
# its cell from a note built out of the draw index, then promotes the entry to the card layer, so a
# chain cut short leaves whole legs behind rather than half a leg.
#
# WAIT FOR QUIET BEFORE TAKING THE LOCK, not after. benchlock guards against other benchlock users
# and nothing else, and on 2026-09-02 the loudest thing on this box was a peer's torch CPU reference
# (esmfold2_e2e_parity at 512 aa, ~7.5 of 8 physical cores) which takes no lock at all. Two of these
# three legs time the whole predict/CLI wall and are roughly half host work, so a co-tenant that
# size lands them low, and low is the direction that hides a regression in a cell with no prior
# value to check it against. benchlock's own 900 s load wait would have warned and measured anyway.
# Waiting outside the lock also avoids holding the box idle against other benchlock users while we
# wait, which is the starvation failure benchlock has no fairness to prevent.
set -u
O=/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap/perf/qb2p300cgap
export BENCHLOCK_LOAD_WAIT_S=2400        # once we hold the lock, still refuse to measure loud
QUIET_MAXLOAD=${QUIET_MAXLOAD:-2.5}
QUIET_DEADLINE=$(( $(date +%s) + ${QUIET_WAIT_S:-28800} ))
{
  echo "=== campaign start $(date -u +%FT%TZ) ==="
  while :; do
    l=$(cut -d' ' -f1 /proc/loadavg)
    n=$(ps -eo args | grep -cE 'esmfold2_e2e_parity|pxd_pagecell|scripts/perf_regression' || true)
    if awk -v a="$l" -v b="$QUIET_MAXLOAD" 'BEGIN{exit !(a+0<=b+0)}' && [ "$n" -le 1 ]; then
      echo "=== box quiet at $(date -u +%FT%TZ), load=$l ==="; break
    fi
    [ "$(date +%s)" -ge "$QUIET_DEADLINE" ] && {
      echo "=== gave up waiting for quiet at $(date -u +%FT%TZ), load=$l cotenants=$n ==="
      echo "=== NOT measuring: a seed cell taken under co-tenant load is a wrong number ==="
      exit 1; }
    sleep 60
  done
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
