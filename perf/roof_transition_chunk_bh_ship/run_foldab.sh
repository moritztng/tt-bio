#!/bin/bash
# Paired fold A/B of the SHIPPED arm: `off` pins the Blackhole row-height raise off on both sides
# of an `on` fold, so the reference is todays behaviour and the test arm is the trees default.
# The flat-height hook is not used -- it cannot express a per-shape height.
set -u
cd "$(dirname "$0")/../.."
CARD="${1:-1}"
exec env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
  TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-bh-ship \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/roof_transition_chunk_bh/foldab.py \
    --out "perf/roof_transition_chunk_bh_ship/out/${3:-foldab}.json" \
    --ref off --legs "${2:-512:on,768:on,1024:on}" --reps "${4:-3}"
