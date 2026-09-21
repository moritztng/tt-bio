#!/usr/bin/env bash
# of3t-tapediverge: the diffusion-module scope, 48 structures, at the 0.4.3 boundary, one arm
# per invocation. Forward and gradient come out of the SAME process, which is the thing D30's
# own warning asks for.
#
#   arms.sh {shipped|shipped2|renorm|selfvalue|sm216|sumall|break}
#
# Every arm writes its own start/end epoch so the AICLK can be read DURING the window.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_refprec/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_tape:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-tapediverge
PY=/home/ttuser/tt-bio-dev/env/bin/python
OUT=/home/ttuser/of3t_tapediverge
mkdir -p "$OUT"
LEV=""
EXTRA=""
RN=0
# TT_BIO_SOFTMAX_BW_RENORM is read at IMPORT time by tt_bio.taped_ttnn. `tdrun.py` has to
# import that module before it can patch it, so the harness's own --softmax-bw-renorm, which
# sets the variable after argparse, lands too late and the arm silently runs shipped. The
# harness's own reach check caught it; the fix is to export the variable before python starts.
case "${1:-}" in
  shipped)   TAG=_td_shipped;   PT=_shipped ;;
  shipped2)  TAG=_td_shipped2;  PT=_shipped2 ;;
  renorm)    TAG=_td_renorm;    PT=_renorm;    RN=1 ;;
  selfvalue) TAG=_td_selfvalue; PT=_selfvalue; RN=1; LEV="sdpa_selfvalue" ;;
  sm216)     TAG=_td_sm216;     PT=_sm216;     RN=1; LEV="sm216_precise" ;;
  sumall)    TAG=_td_sumall;    PT=_sumall;    RN=1; LEV="sum_precise_all" ;;
  break)     TAG=_td_break;     PT=_break;     RN=1; EXTRA="--permute-cot" ;;
  *) echo "usage: arms.sh {shipped|shipped2|renorm|selfvalue|sm216|sumall|break}"; exit 2 ;;
esac
if [ "$RN" = 1 ]; then export TT_BIO_SOFTMAX_BW_RENORM=1; EXTRA="$EXTRA --softmax-bw-renorm"; fi
S=$(date +%s)
echo "=== of3t-tapediverge arm ${1}, 48 structures, card $CARD  $(date -u +%FT%TZ) ==="
echo "ARM_START ${1} $S"
"$PY" perf/of3t_tapediverge/tdrun.py --levers "$LEV" -- \
    perf/of3t_diffusion/device_gradient.py --structs all --tag "$TAG" \
    --cap /home/ttuser/of3t_softgrad/diffcap043 \
    --out-dir perf/of3t_tapediverge \
    --dump-grads "$OUT/device_grads_043all$PT.pt" $EXTRA
rc=$?
E=$(date +%s)
echo "ARM_END ${1} $E  elapsed $((E-S))s"
perf/of3t_f64softmax/clockwin.sh "$CARD" "$S" "$E" || true
echo "TD_ARM_DONE ${1} rc=$rc $(date -u +%FT%TZ)"
exit $rc
