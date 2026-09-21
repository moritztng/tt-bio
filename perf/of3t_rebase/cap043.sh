#!/usr/bin/env bash
# The pairformer block boundary of BUNDLE-MIN-043's own step (D23 deliverable 2).
set -uo pipefail
W=/home/ttuser/of3t_rebase/wt
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
echo "=== capture trunk boundary at 0.4.3  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_gradients/capture_trunk_boundary.py \
    --bundle /home/ttuser/of3t_rebase/bundle_min_043 \
    --manifest-json /home/ttuser/of3t_rebase/bundle_min_043/MANIFEST.json \
    --grads grads_f64_043.pt --blocks 0,23,47 --no-dropout \
    --out /home/ttuser/of3t_rebase/cap043 \
    --report perf/of3t_rebase/capture_trunk_boundary_043.json
echo "=== capture exit $? $(date -u +%FT%TZ) ==="
