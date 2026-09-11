#!/usr/bin/env bash
# b2x-bfp8-pair-track: the cdk2x2_298 control folds and the all-atom/CA RMSD, one process.
set -u
W=/home/ttuser/.coworker/wt/b2x-bfp8-pair-track
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-0}
cd "$W" || exit 1
echo "=== parity card=$CARD $(date -Is) loadavg $(cut -d' ' -f1-3 /proc/loadavg) ==="
env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=0,$CARD \
    TT_BIO_LEASE_HOLDER=worker:b2x-bfp8-pair-track \
    "$PY" -u perf/b2x_bfp8/fold_ab_b8.py --only-parity \
    --out "$W/perf/b2x_bfp8/parity_298_qb2c$CARD.json" \
    --cifdir "$W/perf/b2x_bfp8/cif298"
echo "parity rc=$? $(date -Is)"
