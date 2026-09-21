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
F=tt_bio/openfold3_confidence.py

# arm A: wk/of3t's own file, the campaign baseline
git show origin/wk/of3t:$F > /tmp/conf_wkof3t.py
cp "$F" /tmp/conf_merged.py
cp /tmp/conf_wkof3t.py "$F"
echo "=== arm A (origin/wk/of3t file) $(date -u +%FT%TZ) ==="
timeout 1800 "$PY" perf/of3t_auxfind/default_aa.py /tmp/aa_wkof3t.pt
echo "=== arm A exit $? ==="
git checkout -- "$F"
cmp -s "$F" /tmp/conf_merged.py && echo "RESTORE OK: merged file back in place" || { echo "RESTORE FAILED"; exit 9; }

echo "=== arm B (merged HEAD file) $(date -u +%FT%TZ) ==="
timeout 1800 "$PY" perf/of3t_auxfind/default_aa.py /tmp/aa_merged.pt
echo "=== arm B exit $? ==="
