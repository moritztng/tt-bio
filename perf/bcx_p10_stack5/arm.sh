#!/bin/bash
# One arm of the wave-15 milestone round on a qb2 card: two BindCraft 2 trajectories in one
# process, serial or interleaved through `tt_bio.duotraj`, with every GO lever armed.
#   arm.sh <tag> <rounds> <serial|duo>
#
# `perf/bcx_p10_duotraj/arm.sh` with qb2's interpreter, BindCraft 2 checkout and weights, and
# the full stack on BOTH arms: stack3 (`duo_round.py` hardwires --triatt-bw, --rne-kernel, the
# hifi route, extra-MSA and template), TT_BIO_MM_LAYOUT, TT_BIO_TAPED_CHANNEL_MOVE and
# TT_BIO_WIDEN_ADD. TT_BIO_GRAD_FANIN_L1 stays 0: with widen_add serving every fan-in it has no
# promoted tensor left to place. The arms differ only in serial vs duo.
#
# Nine rounds a trajectory is the ceiling (BindCraft 2's compile path deadlocks at round 10).
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; mode=$3; shift 3
case "$mode" in
    serial) duo=0 ;;
    duo)    duo=1 ;;
    *) echo "arm.sh: mode must be serial or duo, got '$mode'" >&2; exit 2 ;;
esac
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
out=perf/bcx_p10_stack5/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_stack5/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack5
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave "$duo" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
