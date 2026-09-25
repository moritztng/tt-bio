#!/usr/bin/env bash
# Instrument A at aux_heads scope on the 0.4.3 boundary. Card 0 on qb2.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAP=${CAP:-/home/ttuser/of3t_auxheads/cap043}
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
source "$W/perf/refpath.sh"
B=$REF_BUNDLE
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W/perf/of3t_tape" "$W")"
ref_assert "$PY"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-auxheads
echo "=== aux instrument $1 start $(date -u +%FT%TZ) card $CARD ==="
"$PY" perf/of3t_auxheads/aux_instrument.py --boundary "$CAP/boundary_aux_heads.pt" \
    --reference-grads "$B/grads_f64_043.pt" "${@:2}" \
    --out "perf/of3t_auxheads/$1.json"
echo "=== exit $? $(date -u +%FT%TZ) ==="
