#!/bin/bash
# Pass 2, in order of what the campaign needs: first the extent bound that says whether the lever
# exists at a size (contention-immune), then what it is worth in FOLD SECONDS at 512 aa.
#
# Sizes ascend so 768 aa is never the first compile of a session: the cold-768 start in pass 1
# logged a CB set growing past the 1572864 B per-core L1 and then stopped progressing, on the
# SHIPPED arm, which is not this lever and must not be measured into it.
set -u
cd "$(dirname "$0")/../.."
OUT=perf/roof_transition_chunk_bh/out
CARD="${1:-0}"
run() {
  env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
    TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-bh \
    /home/ttuser/tt-bio-dev/env/bin/python3 "$@"
}
run perf/b2z2_size_ladder/ladder.py --levers transition_h --arms ship,h24,h28,h40 \
    --sizes 512,768,1024 --out "$OUT/bound_c$CARD.json" --cifdir "$OUT/cif_bound_c$CARD"
echo "=== bound done $(date -u +%FT%TZ) rc=$?"
run perf/roof_transition_chunk_bh/foldab.py --legs 512:h48 --reps 4 \
    --out "$OUT/foldab_c$CARD.json"
echo "=== foldab done $(date -u +%FT%TZ) rc=$?"
