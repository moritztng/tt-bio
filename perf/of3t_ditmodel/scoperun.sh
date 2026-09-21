#!/usr/bin/env bash
# of3t-ditmodel: re-take the three non-diffusion same-batch scopes on TODAY's tree, at the
# SHIPPED DEFAULT (no flag set), so the model-scope union is one tree rather than four.
#
#   scoperun.sh <cond|aux|msa> <d56on|d56off>
#
# This is of3t-wholemodel's scoperun.sh with two changes: the arms are `d56on` (nothing exported,
# autograd.SOFTMAX_BW_RENORM defaults True) and `d56off` (TT_BIO_SOFTMAX_BW_RENORM=0), and every
# output lands in this row's namespace. The instruments are of3t-direct's and of3t-auxheads',
# unmodified; only the environment differs. armrun.py reads taped_ttnn._SOFTMAX_BW_RENORM back
# out of the loaded module after the run, so which arm ran is a reading and not a label.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
PY=/home/ttuser/tt-bio-dev/env/bin/python
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-ditmodel
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
D=/tmp/of3t/of3t-ditmodel
mkdir -p "$D"
SCOPE=$1; ARM=$2
case "$ARM" in
  d56on)  unset TT_BIO_SOFTMAX_BW_RENORM ;;
  d56off) export TT_BIO_SOFTMAX_BW_RENORM=0 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
case "$SCOPE" in
  cond) export PYTHONPATH="$W/perf/of3t_tape:$W"
        SCRIPT=perf/of3t_conditioning/device_cond_gradient.py
        ARGS=(--tag "_dm_$ARM" --out-dir perf/of3t_ditmodel
              --dump-grads "$D/cond_grads_$ARM.pt") ;;
  aux|msa)
        export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W/perf/of3t_tape" "$W")"
        ref_assert "$PY"
        REF=$REF_BUNDLE/grads_f64_043.pt
        if [ "$SCOPE" = aux ]; then
          SCRIPT=perf/of3t_auxheads/aux_instrument.py
          BND=/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt
        else
          SCRIPT=perf/of3t_auxheads/msa_instrument.py
          BND=/home/ttuser/of3t_auxheads/cap043b/boundary_msa_module.pt
        fi
        ARGS=(--boundary "$BND" --reference-grads "$REF"
              --dump-grads "$D/${SCOPE}_grads_$ARM.pt"
              --out "perf/of3t_ditmodel/instrument_a_${SCOPE}_$ARM.json") ;;
  *) echo "unknown scope $SCOPE"; exit 2 ;;
esac
S=$(date +%s)
echo "=== $SCOPE $ARM start $(date -u +%FT%TZ) card $CARD  renorm_env=${TT_BIO_SOFTMAX_BW_RENORM-<unset>} ==="
"$PY" perf/of3t_wholemodel/armrun.py "$SCRIPT" "${ARGS[@]}"
rc=$?
E=$(date +%s)
perf/of3t_f64softmax/clockwin.sh "$CARD" "$S" "$E" || true
echo "=== $SCOPE $ARM exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
exit $rc
