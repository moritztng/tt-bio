#!/bin/bash
# Correctness half of the shipped Blackhole row-height raise: at 298/512/768/1024 aa, does the
# wired-in per-shape height run, and is the CIF byte-identical to today's shipped output.
# Pass/fail and digests only -- the arms do not share a wall clock, so no ratio comes from here.
set -u
cd "$(dirname "$0")/../.."
CARD="${1:-1}"
exec env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
  TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-bh-ship \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_size_ladder/ladder.py \
    --levers transition_l1 --arms off,on --sizes "${2:-298,512,768,1024}" \
    --out "perf/roof_transition_chunk_bh_ship/out/${3:-ladder}.json" \
    --cifdir "perf/roof_transition_chunk_bh_ship/out/cif_${3:-ladder}"
