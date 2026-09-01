#!/usr/bin/env bash
# campaign.sh — the three card-layer legs the prior pass never got to, card 0, serial.
# Serial on one card on purpose: benchlock is host-scoped, and a concurrent draw on a sibling
# card would contaminate both readings through the shared host dispatch path.
set -u
O=/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed/perf/qb2cardlayer
{
  echo "=== campaign start $(date -u +%FT%TZ) ==="
  bash "$O/draw.sh" esmc-6b 7 0
  bash "$O/draw.sh" boltz2-affinity 5 0
  bash "$O/draw.sh" boltzgen 5 0
  echo "=== campaign done $(date -u +%FT%TZ) ==="
} >> "$O/campaign.out" 2>&1
