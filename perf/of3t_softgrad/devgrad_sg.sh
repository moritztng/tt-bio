#!/usr/bin/env bash
# of3t-softgrad: the four arms at full scope, in this row's own namespace.
#
#   devgrad_sg.sh shipped     CONTROL, must reproduce 7.426217 against upstream's step
#   devgrad_sg.sh precise     precise_config() on the forward softmax        SHIPPABLE
#   devgrad_sg.sh accurate    _accurate_softmax + precise backward reduction SHIPPABLE
#   devgrad_sg.sh sm64        CONTROL, must reproduce 0.0777758
#   devgrad_sg.sh accpermcot  the BREAK control, on a lever arm
#
# Each arm writes its own start/end epoch so the AICLK can be read DURING the timed work
# rather than before it.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_refprec/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_tape:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-softgrad
PY=/home/ttuser/tt-bio-dev/env/bin/python
OUT=/home/ttuser/of3t_softgrad
mkdir -p "$OUT"
case "${1:-}" in
  shipped)    TAG=_sgshipped;   PT=_shipped;   EXTRA="" ;;
  precise)    TAG=_sgprecise;   PT=_precise;   EXTRA="--softmax-lever precise" ;;
  accurate)   TAG=_sgaccurate;  PT=_accurate;  EXTRA="--softmax-lever accurate" ;;
  sm64)       TAG=_sgsm64;      PT=_sm64;      EXTRA="--softmax-f64" ;;
  accpermcot) TAG=_sgaccpermcot; PT=_accpermcot; EXTRA="--softmax-lever accurate --permute-cot" ;;
  *) echo "usage: devgrad_sg.sh {shipped|precise|accurate|sm64|accpermcot}"; exit 2 ;;
esac
S=$(date +%s)
echo "=== of3t-softgrad arm ${1}, 48 structures, tag $TAG, card $CARD  $(date -u +%FT%TZ) ==="
echo "ARM_START ${1} $S"
"$PY" perf/of3t_diffusion/device_gradient.py --structs all --tag "$TAG" \
    --cap /home/ttuser/of3t_softgrad/diffcap043 \
    --out-dir perf/of3t_softgrad \
    --dump-per-tensor $EXTRA \
    --dump-grads "$OUT/device_grads_043all$PT.pt"
rc=$?
E=$(date +%s)
echo "ARM_END ${1} $E  elapsed $((E-S))s"
perf/of3t_residual/clockwin.sh "$S" "$E" || true
echo "DEVGRAD_SG_DONE ${1} rc=$rc $(date -u +%FT%TZ)"
exit $rc
