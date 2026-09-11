#!/usr/bin/env bash
# b2x-bfp8-pair-track: the screen, then the four-arm fold A/B, each under its OWN benchlock
# acquisition so a sibling gets the box back in between rather than being starved across both.
set -u
W=/home/ttuser/.coworker/wt/b2x-bfp8-pair-track
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD="env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:b2x-bfp8-pair-track"

cd "$W" || exit 1
echo "=== screen $(date -Is) ==="
$BL b2x-bfp8-pair-track -- $CARD "$PY" -u perf/b2x_bfp8/screen_block_b8.py \
    --out "$W/perf/b2x_bfp8/screen_block_512_qb2c3.json"
echo "screen rc=$? $(date -Is)"

echo "=== fold A/B $(date -Is) ==="
$BL b2x-bfp8-pair-track -- $CARD "$PY" -u perf/b2x_bfp8/fold_ab_b8.py \
    --out "$W/perf/b2x_bfp8/fold_ab_512_qb2c3.json" --reps 3
echo "fold rc=$? $(date -Is)"
