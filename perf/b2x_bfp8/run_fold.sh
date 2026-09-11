#!/usr/bin/env bash
# b2x-bfp8-pair-track: the four-arm fold A/B. Interleaved A/B/A/B with b16 run every rep as its
# own A/A floor, so the RATIOS are valid co-tenanted (campaign bulletin 2026-09-11 17:20 s4).
# The absolute fold seconds are co-tenanted and are reported as such, not as a headline baseline.
set -u
W=/home/ttuser/.coworker/wt/b2x-bfp8-pair-track
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-0}
REPS=${REPS:-3}
cd "$W" || exit 1
echo "=== fold A/B card=$CARD reps=$REPS $(date -Is) loadavg $(cut -d' ' -f1-3 /proc/loadavg) ==="
env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=0,$CARD \
    TT_BIO_LEASE_HOLDER=worker:b2x-bfp8-pair-track \
    "$PY" -u perf/b2x_bfp8/fold_ab_b8.py \
    --out "$W/perf/b2x_bfp8/fold_ab_512_qb2c$CARD.json" --reps "$REPS"
echo "fold rc=$? $(date -Is)"
