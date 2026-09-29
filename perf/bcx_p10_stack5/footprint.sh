#!/bin/bash
# duotraj's leg 1 re-run on the wave-15 stack: the round's device DRAM floor, banked tape and
# in-seam peak at block granularity. The probe drains the pipeline, so this arm's seconds are
# never quoted. Host RSS comes from the timed arms' own round boundaries.
#   footprint.sh <tag> [rounds]
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=${2:-3}; shift 2 || true
out=perf/bcx_p10_stack5/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_stack5/out/xlacache_fp
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack5
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
export DUOTRAJ_FOOTPRINT_OUT=$PWD/$out/footprint.json
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_duotraj/footprint.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --triatt-bw 1 --rne-kernel 1 \
    --shipped --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
