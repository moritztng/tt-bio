#!/usr/bin/env bash
# The conditioning boundary capture. CPU only -- no device, no card lease.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PY=/home/ttuser/tt-bio-dev/env/bin/python
exec "$PY" perf/of3t_conditioning/capture_cond_boundary.py "$@"
