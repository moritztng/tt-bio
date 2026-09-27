#!/bin/bash
# bcx-p10-mmroof leg 1: cell E, the arithmetic family keyed by shape AND plan, levers ON.
#   cells.sh <out.json> <reps> [extra args...]
# qb2 card 0. The interpreter is bcx_e2e_venv, which is the one that carries torch on this box.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$1; reps=$2; shift 2
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0}
export TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcx-p10-mmroof}
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_mmroof/cells.py \
    --cells E --reps "$reps" --verb-records --out "$out" "$@"
