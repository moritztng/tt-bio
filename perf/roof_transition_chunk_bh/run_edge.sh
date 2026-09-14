#!/bin/bash
# Bracket the Blackhole per-chunk element bound, and record the per-shape height histogram the
# first ladder pass could not (it had no heights census, so "served=2 then a CB clash" did not
# name the shape that threw).
#
# Known from pass 1, h x W for the pair track at c=128: 24576 runs (512@h48, 768@h32),
# 32768 throws (1024@h32), 36864 throws (768@h48). These rungs sit in the gap.
set -u
cd "$(dirname "$0")/../.."
CARD="${1:-3}"
env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
  TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-bh \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_size_ladder/ladder.py \
    --levers transition_h --arms ship,h24,h28,h40 --sizes 768,1024 \
    --out "perf/roof_transition_chunk_bh/out/edge_c$CARD.json" \
    --cifdir "perf/roof_transition_chunk_bh/out/cif_edge_c$CARD"
