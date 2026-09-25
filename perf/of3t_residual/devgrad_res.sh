#!/usr/bin/env bash
# of3t-residual: the same 48-structure device gradient arm of3t-trajectory published, in this
# row's own namespace, plus the new bounds this row adds. Arms:
#
#   devgrad_res.sh sm64        of3t-trajectory's float64-softmax bound, REPRODUCED
#   devgrad_res.sh tr64        the conditioned-transition bound alone
#   devgrad_res.sh sm64tr64    both, which is the arm the row's question is about
#   devgrad_res.sh shipped     the shipped arm, for this row's own baseline
#   devgrad_res.sh permcot     the break control
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$W/perf/of3t_tape" "$W/perf/of3t_gradients" "$W")"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
STRUCTS=all
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-residual
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
OUT=/home/ttuser/of3t_residual
mkdir -p "$OUT"
case "${1:-}" in
  sm64)     TAG=_ressm64;     PT=_sm64;     EXTRA="--softmax-f64" ;;


  permcot)  TAG=_respermcot;  PT=_permcot;  EXTRA="--permute-cot" ;;
  sm64permcot) TAG=_ressm64permcot; PT=_sm64permcot; EXTRA="--softmax-f64 --permute-cot" ;;
  shipped)  TAG=_resshipped;  PT=_shipped;  EXTRA="" ;;
  census)   TAG=_rescensus;   PT=_census;   EXTRA="--verb-census"; STRUCTS=0 ;;
  tr64)     TAG=_restr64b;    PT=_tr64;     EXTRA="--softmax-f64 --host-f64 layer_norm,multiply,multiply_,add,add_" ;;
  tr64lin)  TAG=_restr64lin;  PT=_tr64lin;  EXTRA="--softmax-f64 --host-f64 layer_norm,multiply,multiply_,add,add_,linear" ;;
  *) echo "usage: devgrad_res.sh {sm64|tr64|sm64tr64|permcot|shipped}"; exit 2 ;;
esac
echo "=== device gradient, 48 structures, tag $TAG, card $CARD  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_diffusion/device_gradient.py --structs "$STRUCTS" --tag "$TAG" \
    --cap "$REF_DIFFCAP" \
    --out-dir perf/of3t_residual \
    --dump-per-tensor $EXTRA \
    --dump-grads "$OUT/device_grads_043all$PT.pt"
rc=$?
echo "=== device gradient exit $rc $(date -u +%FT%TZ) ==="
echo "DEVGRAD_RES_DONE rc=$rc $(date -u +%FT%TZ)"
