#!/bin/bash
# The correctness half of the row chunk ladder: does the wider chunk run, wedge or throw at
# 298 / 512 / 768 / 1024 aa, is it bit-exact, and does it actually get SERVED. Pass/fail only.
# No ratio may be quoted from this: the arms do not share a wall clock and qb2 is contended.
set -u
cd "$(dirname "$0")/../.."
exec env TT_VISIBLE_DEVICES="$1" TT_BIO_LEASE_CARDS="$2" \
  TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-bh \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_size_ladder/ladder.py \
    --levers transition_h --arms ship,h32,h48 --sizes "$3" \
    --out "perf/roof_transition_chunk_bh/out/$4.json" \
    --cifdir "perf/roof_transition_chunk_bh/out/cif_$4"
