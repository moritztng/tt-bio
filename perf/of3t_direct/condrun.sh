#!/usr/bin/env bash
# Instrument A at conditioning scope, re-run on wk/of3t-direct for the one thing the published
# run could not do: write the gradient TENSORS, so they can be compared against upstream's own
# bf16 step instead of against a float64 reference upstream never computes (D72).
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
export PYTHONPATH="$W/perf/of3t_tape:$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CARD=${CARD:-1}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-direct
PY=/home/ttuser/tt-bio-dev/env/bin/python
D=/home/ttuser/of3t_direct
mkdir -p "$D"
ARM=${1:-real}; shift || true
case "$ARM" in
  real)    TAG=_fp32        ; EXTRA=() ;;
  revcot)  TAG=_fp32_revcot ; EXTRA=(--negative-control reverse-cot) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
echo "=== cond $ARM start $(date -u +%FT%TZ) card $CARD ==="
"$PY" perf/of3t_conditioning/device_cond_gradient.py \
  --tag "$TAG" --out-dir perf/of3t_direct \
  --dump-grads "$D/cond_grads${TAG}.pt" "${EXTRA[@]}" "$@"
echo "=== exit $? $(date -u +%FT%TZ) ==="
