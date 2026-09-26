#!/bin/bash
# Leg 1, one arm: the composed round's DRAM footprint on pc card 0.
#   footprint.sh <tag> <rounds>
#
# `perf/bcx_p10_stack3/arm.sh` with `footprint.py` in front of `run_round.py` and nothing
# else changed, so the round sampled is the round the campaign gates on. The probe is a
# pipeline drain: this arm's SECONDS are the probe's and are never quoted.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=${2:-4}; shift 2 || true
out=perf/bcx_p10_duotraj/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_duotraj/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-duotraj
export TT_BIO_MM_LAYOUT=1
export TT_BIO_GENQ_COMPACT=0 TT_BIO_TAPED_CHANNEL_MOVE=0
export DUOTRAJ_FOOTPRINT_OUT=$PWD/$out/footprint.json
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_p10_duotraj/footprint.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --triatt-bw 1 --rne-kernel 1 \
    --shipped --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" "$@"
