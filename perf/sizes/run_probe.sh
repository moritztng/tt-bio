#!/usr/bin/env bash
# One capacity/census probe of OpenDDE at a given size on qb1 card 1.
#   run_probe.sh <sizes> <arms> <out.json>
# System interpreter + explicit site-packages: qb1s venv has tt_bio installed editable against the
# shared checkout, and a script run puts scripts/ on sys.path[0], not cwd (memory qb1-editable-install-trap).
set -eu
WT=/home/ttuser/.coworker/wt/opendde-sizes-perf
cd "$WT"
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:opendde-sizes-perf
export PYTHONPATH=/home/ttuser/aiand-bio/env/lib/python3.10/site-packages:$WT
exec /usr/bin/python3.10 perf/other512/fold_ab_multi.py --model opendde \
    --sizes "$1" --arms "$2" --out "$3"
