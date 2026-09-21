#!/usr/bin/env bash
# of3t-nanfloor D2: softmax_accurate re-scored with the chain's reduction config installed,
# and the counterfactual with it at the op default, interleaved with the shipped control.
#
#   arms.sh shipped      CONTROL, must reproduce 7.426217 against upstream's step
#   arms.sh accckc       softmax_accurate, chain reduction at _SOFTMAX_PRECISE_CKC (the published arm)
#   arms.sh accnockc     the same arm with that reduction at the op default
#
# Same boundary, same card, same process shape and the same 48 structures as of3t-softgrad's
# chain.sh, so the numbers sit in that row's table without a conversion.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_refprec/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_tape:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-nanfloor
PY=/home/ttuser/tt-bio-dev/env/bin/python
OUT=/home/ttuser/of3t_nanfloor
mkdir -p "$OUT"
REP=${REP:-1}
case "${1:-}" in
  shipped)  TAG=_nfshipped$REP; PT=_shipped$REP;  EXTRA="" ;;
  accckc)   TAG=_nfaccckc$REP;  PT=_accckc$REP;   EXTRA="--softmax-lever accurate" ;;
  accnockc) TAG=_nfaccnockc$REP; PT=_accnockc$REP; EXTRA="--softmax-lever accurate"
            export OF3T_NANFLOOR_SUM_CKC=none ;;
  *) echo "usage: arms.sh {shipped|accckc|accnockc}"; exit 2 ;;
esac
S=$(date +%s)
echo "=== of3t-nanfloor arm ${1} rep $REP, 48 structures, tag $TAG, card $CARD  $(date -u +%FT%TZ) ==="
echo "ARM_START ${1} $S"
"$PY" perf/of3t_diffusion/device_gradient.py --structs all --tag "$TAG" \
    --cap /home/ttuser/of3t_softgrad/diffcap043 \
    --out-dir perf/of3t_nanfloor \
    --dump-per-tensor $EXTRA \
    --dump-grads "$OUT/device_grads_043all$PT.pt"
rc=$?
E=$(date +%s)
echo "ARM_END ${1} $E  elapsed $((E-S))s"
perf/of3t_residual/clockwin.sh "$S" "$E" || true
echo "ARM_DONE ${1} rc=$rc $(date -u +%FT%TZ)"
exit $rc
