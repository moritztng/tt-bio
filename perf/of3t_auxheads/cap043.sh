#!/usr/bin/env bash
# Capture the 0.4.3 reference boundary for aux_heads, msa_module and input_embedder.
# CPU only, no card: this is upstream float64 against itself. ~25 min at OMP 14.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$W/perf/refpath.sh"
B=$REF_BUNDLE
OUT=${OUT:-/home/ttuser/of3t_auxheads/cap043}
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W")"
ref_assert "$PY"
export OMP_NUM_THREADS=${OMP:-14}
mkdir -p "$OUT"
echo "=== capture start $(date -u +%FT%TZ)  OMP=$OMP_NUM_THREADS ==="
nice -n 15 "$PY" perf/of3t_auxheads/capture_boundary.py \
    --batch "$B/batch_step003.pt" \
    --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
    --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --replay-draws "$B/draws_recycles0.pt" \
    --reference-grads "$B/grads_f64_043.pt" \
    --num-recycles 0 --out "$OUT"
echo "=== capture exit $? $(date -u +%FT%TZ) ==="
echo "CAP043_ALLDONE $(date -u +%FT%TZ)"
