#!/usr/bin/env bash
# The perf and UX legs of RELEASING.md on the merged tree. UX first: it is plumbing, not timing, so
# it can share a box with the parity gate. perf_regression cannot -- it refuses a contended host
# unless told otherwise, and a 15% band on a 20 aa fold is exactly what host load eats.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-chunk-remerge-verify
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=$WT/perf/roof_transition_chunk_bh_ship/out
cd "$WT" || exit 1
CARD=${CARD:-2}
case "${1:-ux}" in
ux)
  exec env PYTHONPATH="$WT" TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
    TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-remerge-verify \
    OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt \
    OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3 \
    "$PY" scripts/ux_regression.py >> "$O/ux_remerge.log" 2>&1
  ;;
perf)
  # benchlock waits for the lock AND for the box to go quiet, which is the whole point here: the
  # baseline cells are 15%-banded warm medians and co-tenant noise on qb2 is 1-10%.
  exec env PYTHONPATH="$WT" TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
    TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-remerge-verify \
    bash "$HOME/.coworker/scripts/benchlock.sh" roof-transition-chunk-remerge-verify -- \
    "$PY" scripts/perf_regression.py >> "$O/perf_remerge.log" 2>&1
  ;;
esac
