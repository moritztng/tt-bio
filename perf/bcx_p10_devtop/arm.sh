#!/bin/bash
# One arm of the bcx-p10-devtop sitting on qb2 card 0: N=3 interleaved BindCraft 2 trajectories
# on the full tritraj stack, with or without TT_BIO_QKV_GRAD_JOIN.
#   arm.sh <tag> <rounds> <qkv_join 0|1> [duo_round.py args...]
# `perf/bcx_p10_tritraj/arm.sh` at N=3 inter; the arms differ only in the one gate.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; join=$3; shift 3
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
out=perf/bcx_p10_devtop/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_devtop/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-devtop
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
export TT_BIO_QKV_GRAD_JOIN=$join
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave 1 --trajectories 3 --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
