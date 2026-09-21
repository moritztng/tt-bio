#!/usr/bin/env bash
# Instrument A at aux_heads / msa_module scope, re-run on wk/of3t-direct for the one thing the
# published runs could not do: write the gradient TENSORS, so they can be scored against
# upstream's own bf16 step instead of only against a float64 reference upstream never computes.
#   $1 = aux|msa    $2 = real|scramble
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W/perf/of3t_tape" "$W")"
ref_assert "$PY"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CARD=${CARD:-1}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-direct
REF=$REF_BUNDLE/grads_f64_043.pt
D=/home/ttuser/of3t_direct
mkdir -p "$D"
case "$1" in
  aux) SCRIPT=perf/of3t_auxheads/aux_instrument.py
       BND=/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt ;;
  msa) SCRIPT=perf/of3t_auxheads/msa_instrument.py
       BND=/home/ttuser/of3t_auxheads/cap043b/boundary_msa_module.pt ;;
  *) echo "unknown scope $1"; exit 2 ;;
esac
case "$2" in
  real)     TAG=""        ; EXTRA=() ;;
  scramble) TAG="_scramcot"; EXTRA=(--scramble-cot) ;;
  *) echo "unknown arm $2"; exit 2 ;;
esac
echo "=== $1 $2 start $(date -u +%FT%TZ) card $CARD ==="
"$PY" "$SCRIPT" --boundary "$BND" --reference-grads "$REF" \
  --dump-grads "$D/${1}_grads${TAG}.pt" \
  --out "perf/of3t_direct/instrument_a_043_$1${TAG}.json" "${EXTRA[@]}"
echo "=== exit $? $(date -u +%FT%TZ) ==="
