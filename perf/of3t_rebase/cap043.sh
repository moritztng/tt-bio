#!/usr/bin/env bash
# The pairformer block boundary of BUNDLE-MIN-043's own step (D23 deliverable 2).
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$REF_PYLIBS")"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
echo "=== capture trunk boundary at 0.4.3  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_gradients/capture_trunk_boundary.py \
    --bundle "$REF_BUNDLE" \
    --manifest-json "$REF_BUNDLE/MANIFEST.json" \
    --grads grads_f64_043.pt --blocks 0,23,47 --no-dropout \
    --out "$REF_CAP" \
    --report perf/of3t_rebase/capture_trunk_boundary_043.json
echo "=== capture exit $? $(date -u +%FT%TZ) ==="
