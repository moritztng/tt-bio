#!/bin/bash
# bcx-p10-mmlay leg 1: cell E, 3 reps, the matmul family keyed by shape and placement.
#   cells.sh <out.json> <reps> [extra args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$1; reps=$2; shift 2
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-mmlay
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_p10_mmlay/cells.py \
    --cells E --reps "$reps" --verb-records --out "$out" "$@"
