#!/bin/bash
# bcx-p10-calls: `perf/bcx_p10_shape/shape.py` cell E -- the fold's own program at 288 with the
# hifi route -- with `--verb-records`, which is the only thing this row adds to it. Cell E is
# the anchor `bcx-p10-devgap` attributed; no other cell is needed to census op names.
#   cells.sh <out.json> <reps> [extra args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$1; reps=$2; shift 2
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-calls
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_p10_shape/shape.py cells \
    --cells E --reps "$reps" --verb-records --out "$out" "$@"
