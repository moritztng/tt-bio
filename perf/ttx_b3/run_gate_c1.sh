#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/ttx-b3-pairffn-fc1-l1-ship
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1        TT_BIO_LEASE_HOLDER=worker:ttx-b3-pairffn-fc1-l1-ship PYTHONPATH="$WT"
echo "=== gate start $(date -Is) ==="
"$PY" -u scripts/full_parity_gate.py --workers tt-quietbox2:1 --fresh   --leg esmfold2-trpcage --leg esmfold2-fast-trpcage --leg esmfold2-cocrystal   --workdir "$WT/perf/ttx_b3/gate_work" --out perf/ttx_b3/gate_esmfold2_c1.json
echo "=== gate rc=$? end $(date -Is) ==="
