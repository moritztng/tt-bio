#!/usr/bin/env bash
# of3t-ditmodel: re-take the diffusion arm that MODEL_shipped.json's diffusion_transformer
# section was scored on, on today's tree, with D56 explicitly ON and explicitly OFF.
#
#   run_arm.sh d56on    the SHIPPED DEFAULT. TT_BIO_SOFTMAX_BW_RENORM is left unset so the arm
#                       is the composition a user gets, not a flag this harness turned on.
#                       autograd.SOFTMAX_BW_RENORM defaults True since 2026-09-21 (ask 9629).
#   run_arm.sh d56off   the BREAK CONTROL, TT_BIO_SOFTMAX_BW_RENORM=0, everything else identical.
#                       Predicted to bring MODEL_shipped's 8.1943 back.
#
# The capture, the checkpoint, the boundary and the cotangent are `of3t-f64softmax`'s and
# `of3t-ditref`'s: /home/ttuser/of3t_softgrad/diffcap043, stamped 0.4.3 with 761 parameters and
# 24 per-block attention_pair_bias.layer_norm_z. device_gradient.py resolves the reference tree
# with refpath.assert_resolved() IN THIS PROCESS and refuses a capture stamped against another.
#
# Card 0 by grant. AICLK sampled DURING the window from qbcard/cardtel.tsv, never before it.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_refprec/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_tape:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-ditmodel
PY=/home/ttuser/tt-bio-dev/env/bin/python
OUT=/tmp/of3t/of3t-ditmodel
mkdir -p "$OUT"
case "${1:-}" in
  d56on)  TAG=_d56on;  unset TT_BIO_SOFTMAX_BW_RENORM ;;
  d56off) TAG=_d56off; export TT_BIO_SOFTMAX_BW_RENORM=0 ;;
  *) echo "usage: run_arm.sh {d56on|d56off}"; exit 2 ;;
esac
S=$(date +%s)
echo "=== of3t-ditmodel arm ${1}, 48 structures, tag $TAG, card $CARD  $(date -u +%FT%TZ) ==="
echo "ENV TT_BIO_SOFTMAX_BW_RENORM=${TT_BIO_SOFTMAX_BW_RENORM-<unset>}"
echo "ARM_START ${1} $S  load $(cut -d' ' -f1-3 /proc/loadavg)"
"$PY" perf/of3t_diffusion/device_gradient.py --structs all --tag "$TAG" \
    --cap /home/ttuser/of3t_softgrad/diffcap043 \
    --out-dir perf/of3t_ditmodel \
    --dump-per-tensor \
    --dump-grads "$OUT/device_grads_043all${TAG}.pt"
rc=$?
E=$(date +%s)
echo "ARM_END ${1} $E  elapsed $((E-S))s  load $(cut -d' ' -f1-3 /proc/loadavg)"
perf/of3t_f64softmax/clockwin.sh "$CARD" "$S" "$E" || true
echo "DITMODEL_ARM_DONE ${1} rc=$rc $(date -u +%FT%TZ)"
exit $rc
