#!/usr/bin/env bash
# Release gate, perf + UX legs.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
echo "########## perf_regression ##########"
$PY -u scripts/perf_regression.py
echo "PERF_RC=$?"
echo "########## ux_regression ##########"
$PY -u scripts/ux_regression.py
echo "UX_RC=$?"
echo "PERFUX_DONE"
