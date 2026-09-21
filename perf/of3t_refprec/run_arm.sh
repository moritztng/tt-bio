#!/usr/bin/env bash
# One precision arm of OF3T row of3t-refprec. Everything except --dtype/--autocast is pinned to
# the float64 reference's own inputs, so an arm's difference from it is precision and nothing else.
set -uo pipefail
R=/home/ttuser/of3t_refprec
WT=/home/ttuser/.coworker/wt/of3t-refprec
PY=/home/ttuser/tt-bio-dev/env/bin/python
export PYTHONPATH="$R/of3pkg043:$R/deps:$R/pylibs"
export OMP_NUM_THREADS="${OMP:-7}"
BATCH_SHA=3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f

name="$1"; dtype="$2"; ac="$3"; draws="${4:-$R/bundle_ref/draws_recycles0.pt}"
echo "=== arm $name dtype=$dtype autocast=$ac draws=$(basename "$draws") OMP=$OMP_NUM_THREADS $(date -u +%FT%TZ) ==="
nice -n 10 "$PY" "$WT/perf/of3t_reference/bundle_min.py" \
    --batch "$R/bundle_ref/batch_step003.pt" --batch-sha256 "$BATCH_SHA" \
    --out "$R/run/$name" --dtype "$dtype" --autocast "$ac" --num-recycles 0 \
    --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --replay-draws "$draws" --fd-samples 0
echo "=== arm $name exit $? $(date -u +%FT%TZ) ==="
