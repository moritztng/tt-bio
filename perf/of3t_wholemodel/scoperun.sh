#!/usr/bin/env bash
# of3t-wholemodel: re-take the three same-batch scopes of3t-direct measured, under this row's
# repair arms. The instruments are of3t-direct's, unmodified; only the environment differs.
#   scoperun.sh <cond|aux|msa> <shipped|renorm|renormf64>
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-wholemodel
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
D=/home/ttuser/of3t_wholemodel
mkdir -p "$D"
SCOPE=$1; ARM=$2
case "$ARM" in
  shipped)   ;;
  renorm)    export TT_BIO_SOFTMAX_BW_RENORM=1 ;;
  renormf64) export TT_BIO_SOFTMAX_BW_RENORM=1; export TT_BIO_HOST_F64_SOFTMAX_AB=all ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
case "$SCOPE" in
  cond) export PYTHONPATH="$W/perf/of3t_tape:$W"
        SCRIPT=perf/of3t_conditioning/device_cond_gradient.py
        ARGS=(--tag "_wm_$ARM" --out-dir perf/of3t_wholemodel
              --dump-grads "$D/cond_grads_$ARM.pt") ;;
  aux|msa)
        export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs:$W/perf/of3t_tape:$W"
        REF=/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt
        if [ "$SCOPE" = aux ]; then
          SCRIPT=perf/of3t_auxheads/aux_instrument.py
          BND=/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt
        else
          SCRIPT=perf/of3t_auxheads/msa_instrument.py
          BND=/home/ttuser/of3t_auxheads/cap043b/boundary_msa_module.pt
        fi
        ARGS=(--boundary "$BND" --reference-grads "$REF"
              --dump-grads "$D/${SCOPE}_grads_$ARM.pt"
              --out "perf/of3t_wholemodel/instrument_a_${SCOPE}_$ARM.json") ;;
  *) echo "unknown scope $SCOPE"; exit 2 ;;
esac
S=$(date +%s)
echo "=== $SCOPE $ARM start $(date -u +%FT%TZ) card $CARD ==="
"$PY" perf/of3t_wholemodel/armrun.py "$SCRIPT" "${ARGS[@]}"
rc=$?
E=$(date +%s)
echo "=== $SCOPE $ARM exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
exit $rc
