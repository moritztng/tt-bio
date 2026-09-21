#!/usr/bin/env bash
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W")"
ref_assert "$PY"
export OMP_NUM_THREADS=12
echo "=== arm_p $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_auxfind/arm_p.py --out perf/of3t_auxfind/arm_p.json --threads 12
echo "=== exit $? $(date -u +%FT%TZ) ==="
