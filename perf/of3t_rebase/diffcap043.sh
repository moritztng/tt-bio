#!/usr/bin/env bash
# The diffusion-module boundary of BUNDLE-MIN-043's own step: A18's gate (deliverable 4).
#
# A18's discriminator scores our xl_out against S["xl_out"] in sub_boundary.pt
# (device_gradient.py:316). That capture was taken through a 0.5.0 forward, so unlike the DiT
# bisection -- which hands both sides the same captured inputs and needed only a PYTHONPATH
# change -- it cannot be re-read at 0.4.3 until the boundary is re-captured there.
set -uo pipefail
W=/home/ttuser/of3t_rebase/wt
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
B=/home/ttuser/of3t_rebase/bundle_min_043
C=/home/ttuser/of3t_rebase/diffcap043

echo "=== diffusion boundary at 0.4.3  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_diffusion/capture_diffusion_boundary.py \
    --bundle "$B" --manifest-json "$B/MANIFEST.json" \
    --grads grads_f64_043.pt --reference-key validated_gradient \
    --out "$C" --report perf/of3t_rebase/capture_diffusion_boundary_043.json
echo "=== diffusion boundary exit $? $(date -u +%FT%TZ) ==="

echo "=== sub boundary at 0.4.3  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_diffusion/sub_boundary.py \
    --boundary "$C/diffusion_boundary.pt" --out "$C/sub_boundary.pt" \
    --report perf/of3t_rebase/sub_boundary_043.json
echo "=== sub boundary exit $? $(date -u +%FT%TZ) ==="
echo "DIFFCAP_ALLDONE $(date -u +%FT%TZ)"
