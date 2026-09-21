#!/usr/bin/env bash
# Instrument A at aux_heads scope, re-taken with of3t-auxfind's mask fix on. qb2, card $CARD.
# Usage: gradrun.sh <out-stem> [extra args...]
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-auxgrad
CAP=${CAP:-/home/ttuser/of3t_auxheads/cap043}
B=/home/ttuser/of3t_rebase/bundle_min_043
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs:$W/perf/of3t_confidence:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-1}
# card 1 is this row's grant; a sibling card is fanned out to only when it is genuinely idle,
# and the lease is widened on that command rather than moved off the grant.
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=1,$CARD TT_BIO_LEASE_HOLDER=worker:of3t-auxgrad
echo "=== auxgrad $1 start $(date -u +%FT%TZ) card $CARD ==="
"$PY" perf/of3t_auxgrad/auxgrad_instrument.py --boundary "$CAP/boundary_aux_heads.pt" \
    --reference-grads "$B/grads_f64_043.pt" "${@:2}" \
    --out "perf/of3t_auxgrad/$1.json"
echo "=== exit $? $(date -u +%FT%TZ) ==="
