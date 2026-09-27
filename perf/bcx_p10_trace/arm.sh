#!/bin/bash
# One arm of the trace sitting on a qb2 card: N BindCraft 2 trajectories in one process, eager
# or with the taped seams replayed from ttnn traces (TT_BIO_TRACE_SEAMS), full stack5 levers.
#   arm.sh <tag> <rounds> <N> <inter|serial> <trace 0|1> [duo_round.py args...]
#
# `perf/bcx_p10_tritraj/arm.sh` plus the one gate. The arms differ only in it.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; n=$3; mode=$4; trace=$5; shift 5
case "$mode" in
    serial) inter=0 ;;
    inter)  inter=1 ;;
    *) echo "arm.sh: mode must be inter or serial, got '$mode'" >&2; exit 2 ;;
esac
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
out=perf/bcx_p10_trace/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_trace/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-trace
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
export TT_BIO_TRACE_SEAMS=$trace
exec timeout 2400 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave "$inter" --trajectories "$n" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
