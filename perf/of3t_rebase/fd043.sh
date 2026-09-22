#!/usr/bin/env bash
# The float64 finite-difference validation of BUNDLE-MIN-043, sampled by gradient mass.
#
# Standing rule: a gradient is validated against a float64 reference that is itself validated by
# float64 central finite differences. A13 says two runs of the reference agree; this says the
# reference is the derivative of the forward it claims to differentiate. The 0.5.0 bundle's FD
# figures cannot be inherited, because the revision changed the function.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=${OF3T_REBASE_RUN:-$HOME/of3t_rebase_run}
B=/home/ttuser/of3t/bundle_min
PY=/home/ttuser/tt-bio-dev/env/bin/python
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS")"
ref_assert "$PY"
mkdir -p "$R/run"
export OMP_NUM_THREADS=12
BATCH_SHA=3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f
cd "$W"
echo "=== FD validation of BUNDLE-MIN-043, sampled by gradient mass  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_reference/bundle_min.py \
    --batch "$B/batch_step003.pt" --batch-sha256 "$BATCH_SHA" \
    --out "$R/run/out_043_fd" --dtype float64 --num-recycles 0 \
    --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --replay-draws "$B/draws_recycles0.pt" --fd-samples 8 --fd-h 1e-4 --fd-sample-by norm
echo "=== FD exit $? $(date -u +%FT%TZ) ==="
echo "FD_ALLDONE $(date -u +%FT%TZ)"
