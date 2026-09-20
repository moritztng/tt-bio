#!/usr/bin/env bash
# of3t-f64softmax: the supported host float64 softmax path, scored at full scope on card 3.
#
#   devgrad_f64.sh shipped     CONTROL, must reproduce 7.426217e+00 against upstream's step
#   devgrad_f64.sh sitef64     the CODE PATH, selected by TT_BIO_HOST_F64_SOFTMAX_AB
#   devgrad_f64.sh sitef64pc   the BREAK control, the same path with structure k+1's cotangent
#
# of3t-softgrad measured the same bound as a rule installed over the tape verb and read
# 7.777580e-02. This row measures the shipped call sites with the site flag on. The two are
# different constructions of the same arithmetic, so shipped reproducing 7.426217 and sitef64
# landing on softgrad's number is the instrument floor for the whole row.
#
# Each arm writes its own start/end epoch so the AICLK is read DURING the timed work.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_refprec/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_tape:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-3}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-f64softmax
PY=/home/ttuser/tt-bio-dev/env/bin/python
OUT=/home/ttuser/of3t_f64softmax
mkdir -p "$OUT"
case "${1:-}" in
  shipped)   TAG=_f64shipped; PT=_shipped; EXTRA="" ;;
  sitef64)   TAG=_f64site;    PT=_sitef64; EXTRA="--softmax-site-f64" ;;
  sitef64pc) TAG=_f64sitepc;  PT=_sitepc;  EXTRA="--softmax-site-f64 --permute-cot" ;;
  *) echo "usage: devgrad_f64.sh {shipped|sitef64|sitef64pc}"; exit 2 ;;
esac
S=$(date +%s)
echo "=== of3t-f64softmax arm ${1}, 48 structures, tag $TAG, card $CARD  $(date -u +%FT%TZ) ==="
echo "ARM_START ${1} $S"
"$PY" perf/of3t_diffusion/device_gradient.py --structs all --tag "$TAG" \
    --cap /home/ttuser/of3t_softgrad/diffcap043 \
    --out-dir perf/of3t_f64softmax \
    --dump-per-tensor $EXTRA \
    --dump-grads "$OUT/device_grads_043all$PT.pt"
rc=$?
E=$(date +%s)
echo "ARM_END ${1} $E  elapsed $((E-S))s"
perf/of3t_f64softmax/clockwin.sh "$CARD" "$S" "$E" || true
echo "DEVGRAD_F64_DONE ${1} rc=$rc $(date -u +%FT%TZ)"
exit $rc
