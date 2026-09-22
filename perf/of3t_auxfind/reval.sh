#!/usr/bin/env bash
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W")"
ref_assert "$PY"
export OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-auxfind
echo "=== port_vs_arms on merged HEAD, card 0 $(date -u +%FT%TZ) ==="
timeout 2400 "$PY" perf/of3t_auxfind/port_vs_arms.py --out perf/of3t_auxfind/port_vs_arms_merged.json --threads 8
echo "=== exit $? $(date -u +%FT%TZ) ==="
