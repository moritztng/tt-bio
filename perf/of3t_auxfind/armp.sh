#!/usr/bin/env bash
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-auxfind
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs:$W"
export OMP_NUM_THREADS=12
echo "=== arm_p $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_auxfind/arm_p.py --out perf/of3t_auxfind/arm_p.json --threads 12
echo "=== exit $? $(date -u +%FT%TZ) ==="
