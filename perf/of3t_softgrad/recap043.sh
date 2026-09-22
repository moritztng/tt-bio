#!/usr/bin/env bash
# of3t-softgrad: rebuild the 0.4.3 diffusion boundary.
#
# `/home/ttuser/of3t_rebase/` was pruned with its row's worktree, taking `diffcap043` with it.
# Every arm this row runs scores against that capture, so it is rebuilt here, in this row's own
# namespace under $HOME rather than inside a worktree that can be pruned the same way.
#
# The bundle it is built from survives: `/home/ttuser/of3t_refprec/bundle_ref` IS
# `bundle_min_043` -- its own MANIFEST.json names `/home/ttuser/of3t_rebase/bundle_min_043` as
# its canonical location, and the five declared hashes are the five the original capture
# verified.
#
# The rebuild is CHECKED, not assumed: `capture_diffusion_boundary_043.json` recorded
# loss 1.2675874205688995, cot_norm 0.019426651390714835 and vs_bundle worst_rel 0.0. If this
# capture is the same function of the same bundle it reproduces all three to every digit, and
# OMP_NUM_THREADS is pinned at the 8 the original used because a float64 CPU reduction is a
# function of its thread count.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$W/perf/of3t_gradients" "$W")"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
B=$REF_BUNDLE
C=$REF_DIFFCAP
mkdir -p "$C"

echo "=== diffusion boundary at 0.4.3, rebuilt  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_diffusion/capture_diffusion_boundary.py \
    --bundle "$B" --manifest-json "$B/MANIFEST.json" \
    --grads grads_f64_043.pt --reference-key validated_gradient \
    --out "$C" --report perf/of3t_softgrad/capture_diffusion_boundary_043_REBUILT.json
echo "=== diffusion boundary exit $? $(date -u +%FT%TZ) ==="

echo "=== sub boundary at 0.4.3  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_diffusion/sub_boundary.py \
    --boundary "$C/diffusion_boundary.pt" --out "$C/sub_boundary.pt" \
    --report perf/of3t_softgrad/sub_boundary_043_REBUILT.json
echo "=== sub boundary exit $? $(date -u +%FT%TZ) ==="
echo "RECAP043_DONE $(date -u +%FT%TZ)"
