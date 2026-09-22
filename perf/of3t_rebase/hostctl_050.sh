#!/usr/bin/env bash
# The host control for the 0.4.3 rebuild.
#
# The published bundle was produced on a rented A100 under CUDA; the 0.4.3 rebuild is produced on
# qb2 CPU. So the rebuild moves the upstream revision AND the box at the same time, and the
# campaign's own capture already shows the box is not free: capture_trunk_boundary on qb2 CPU
# reproduced the A100 bundle's loss only to 1.686e-02 and its global gradient norm to 4.617e-02.
#
# This arm is upstream 0.5.0, the published revision, rebuilt on qb2 CPU with the same batch, the
# same replayed draws and the same r = 0. Against the published A100 bundle it measures the BOX.
# Against the 0.4.3 rebuild, box held fixed, it measures the REVISION. Without it every trunk
# number downstream has two moving parts, which is the mistake this row exists to undo.
#
# --allow-unexpected because 0.5.0 cannot load these weights cleanly. That is the finding, and
# producing the control deliberately is the one use the escape hatch has.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=${OF3T_REBASE_RUN:-$HOME/of3t_rebase_run}
B=/home/ttuser/of3t/bundle_min
PY=/home/ttuser/tt-bio-dev/env/bin/python
source "$W/perf/refpath.sh"
# This arm is upstream 0.5.0 ON PURPOSE -- it holds the box fixed and moves the revision -- so
# it pins and asserts 0.5.0, not the campaign's 0.4.3 default. It used of3t_gradients/of3pkg,
# which is the same 0.5.0 tree (092fb575...) as of3pkg050 under the restored references.
export PYTHONPATH="$REF_OF3PKG050:$REF_DEPS:$REF_PYLIBS"
ref_assert "$PY" "$REF_OF3PKG050"
mkdir -p "$R/run"
export OMP_NUM_THREADS=8
BATCH_SHA=3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f
cd "$W"
echo "=== 0.5.0 on qb2 CPU, host control  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_reference/bundle_min.py \
    --batch "$B/batch_step003.pt" --batch-sha256 "$BATCH_SHA" \
    --out "$R/run/out_050_hostctl" --dtype float64 --num-recycles 0 \
    --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --replay-draws "$B/draws_recycles0.pt" --fd-samples 0 --allow-unexpected
echo "=== host control exit $? $(date -u +%FT%TZ) ==="
