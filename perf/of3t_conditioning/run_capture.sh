#!/usr/bin/env bash
# The conditioning boundary capture. CPU only -- no device, no card lease.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$W/perf/of3t_gradients" "$W")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
exec "$PY" perf/of3t_conditioning/capture_cond_boundary.py "$@"
