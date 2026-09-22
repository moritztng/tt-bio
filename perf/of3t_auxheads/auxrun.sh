#!/usr/bin/env bash
# Instrument A at aux_heads scope on the 0.4.3 boundary. Card 0 on qb2.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-auxheads
CAP=${CAP:-/home/ttuser/of3t_auxheads/cap043}
B=/home/ttuser/of3t_rebase/bundle_min_043
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs:$W/perf/of3t_tape:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-auxheads
echo "=== aux instrument $1 start $(date -u +%FT%TZ) card $CARD ==="
"$PY" perf/of3t_auxheads/aux_instrument.py --boundary "$CAP/boundary_aux_heads.pt" \
    --reference-grads "$B/grads_f64_043.pt" "${@:2}" \
    --out "perf/of3t_auxheads/$1.json"
echo "=== exit $? $(date -u +%FT%TZ) ==="
