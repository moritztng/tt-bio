#!/usr/bin/env bash
# b2x-bfp8-pair-track: the block screen. Falsifier (b) of the precision axis.
# No benchlock: block-level A/B is interleaved with its own A/A floor, which the campaign's
# 2026-09-11 17:20 bulletin rules measurable co-tenanted.
set -u
W=/home/ttuser/.coworker/wt/b2x-bfp8-pair-track
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-1}
cd "$W" || exit 1
echo "=== screen card=$CARD $(date -Is) loadavg $(cut -d' ' -f1-3 /proc/loadavg) ==="
env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=0,$CARD \
    TT_BIO_LEASE_HOLDER=worker:b2x-bfp8-pair-track \
    "$PY" -u perf/b2x_bfp8/screen_block_b8.py \
    --out "$W/perf/b2x_bfp8/screen_block_512_qb2c$CARD.json" --reps 5
echo "screen rc=$? $(date -Is)"
