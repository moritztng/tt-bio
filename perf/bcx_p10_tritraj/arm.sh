#!/bin/bash
# One arm of the wave-16 tritraj sitting on a qb2 card: N BindCraft 2 trajectories in one
# process, interleaved through `tt_bio.duotraj` (or serial, for the leg-4 reference), with the
# full stack5 configuration on every arm.
#   arm.sh <tag> <rounds> <N> <inter|serial> [duo_round.py args...]
#
# `perf/bcx_p10_stack5/arm.sh` with `--trajectories N`. The arms differ only in N.
# XLA_FLAGS passes through, for the host-pool-capped pair.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; n=$3; mode=$4; shift 4
case "$mode" in
    serial) inter=0 ;;
    inter)  inter=1 ;;
    *) echo "arm.sh: mode must be inter or serial, got '$mode'" >&2; exit 2 ;;
esac
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
out=perf/bcx_p10_tritraj/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_tritraj/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-tritraj
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave "$inter" --trajectories "$n" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
