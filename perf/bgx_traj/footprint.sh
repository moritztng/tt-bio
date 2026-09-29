#!/bin/bash
# One rung of the bgx-traj device ladder: what ONE shipped BindCraft 2 trajectory holds on the
# card at a given token axis. `perf/bcx_p10_duotraj/footprint.py` unchanged underneath, on qb2
# paths, with the binder length as the size knob: the PD-L1 target contributes 129 tokens, so
# the axis is pad32(binder + 129).
#   footprint.sh <tag> <binder> [rounds]
# The probe drains the pipeline at every block, so this arms SECONDS are the probes and are
# never quoted as a round time.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; binder=$2; rounds=${3:-3}; shift 3 || shift 2
out=perf/bgx_traj/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_traj/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=worker:bgx-traj
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export DUOTRAJ_FOOTPRINT_OUT=$PWD/$out/footprint.json
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_duotraj/footprint.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --triatt-bw 1 --rne-kernel 1 \
    --shipped --binder "$binder" \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
