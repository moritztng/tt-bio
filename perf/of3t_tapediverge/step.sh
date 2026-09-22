#!/usr/bin/env bash
# of3t-tapediverge: D32's second consequence, measured. One host, one card, one structure, crop
# 384, batch 1 -- the taped arm and the untaped arm through the SAME harness, so the
# taped-vs-untaped ratio is measured rather than asserted.
#
#   step.sh {taped|notape}
#
# `perf/of3t_perf/step.py` intercepts a real `predict_one`, so the input is the shipped
# pipeline's own featurisation and there is no second one to drift. The AICLK is sampled inside
# the harness DURING the timed work and again here over the whole window, independently.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
export PYTHONPATH="$W/perf/of3t_tape:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-tapediverge
PY=/home/ttuser/tt-bio-dev/env/bin/python
case "${1:-}" in
  taped)  EXTRA=""          ; OUTF=perf/of3t_tapediverge/STEP_384_taped.json ;;
  notape) EXTRA="--no-tape" ; OUTF=perf/of3t_tapediverge/STEP_384_notape.json ;;
  *) echo "usage: step.sh {taped|notape}"; exit 2 ;;
esac
S=$(date +%s)
echo "=== of3t-tapediverge step ${1}, crop 384, batch 1, card $CARD  $(date -u +%FT%TZ) ==="
echo "STEP_START ${1} $S"
"$PY" perf/of3t_perf/step.py --tokens 384 --reps 3 --out "$OUTF" $EXTRA
rc=$?
E=$(date +%s)
echo "STEP_END ${1} $E  elapsed $((E-S))s"
perf/of3t_f64softmax/clockwin.sh "$CARD" "$S" "$E" || true
echo "TD_STEP_DONE ${1} rc=$rc $(date -u +%FT%TZ)"
exit $rc
