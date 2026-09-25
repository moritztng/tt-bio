#!/usr/bin/env bash
# Block-boundary ladder captures (D34): where does block 47's 1.18x inflation turn on?
#
# The per-block arms ARE the block-boundary cut D34 asks for -- each is driven by the bundle's
# OWN captured cotangent at that block, so the cotangent entering is the reference's byte for
# byte and any inflation is generated inside the block. Blocks 0 and 23 read norm ratio 1.000
# and block 47 reads 1.18. This captures the intermediate boundaries so the turn-on can be
# located instead of bracketed 24 blocks wide.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$REF_PYLIBS")"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
B=$REF_BUNDLE
echo "=== block-boundary ladder capture at 0.4.3  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_gradients/capture_trunk_boundary.py \
    --bundle "$B" --manifest-json "$B/MANIFEST.json" --grads grads_f64_043.pt \
    --blocks 0,8,16,23,32,40,47 --no-dropout \
    --out "$REF_CAP_LADDER" \
    --report perf/of3t_rebase/capture_trunk_boundary_043_ladder.json
echo "=== capture exit $? $(date -u +%FT%TZ) ==="
echo "CAPLADDER_ALLDONE $(date -u +%FT%TZ)"
