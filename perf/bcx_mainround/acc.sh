#!/bin/bash
# bcx-mainround accuracy and reach legs, after the timed sitting, same card, serially.
#   acc.sh [legs]   legs: vjp,count (default both)
# vjp: `perf/bcx_p10_stack2/grad_stack2.py --arm on` in each tree with the product's fast-round
# levers from the environment (stack5's STACK plus the QKV join, softmax backward bf16 as
# `fast_round` keeps it). count: arm B's round under `count_ln.py`, 3 rounds, N=3.
set -uo pipefail
B=$(cd "$(dirname "$0")/../.." && pwd); A=$B/.armA
legs=${1:-vjp,count}
o=$B/perf/bcx_mainround/out; mkdir -p "$o"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-mainround
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
PY=/home/ttuser/bcx_e2e_venv/bin/python3
STACK="TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1 TT_BIO_GRAD_FANIN_L1=0 TT_BIO_QKV_GRAD_JOIN=1 TT_BIO_SOFTMAX_BW_FP32=0"
[ "${legs/vjp/}" = "$legs" ] || for arm in A B; do
    case $arm in A) t=$A;; B) t=$B;; esac
    echo "=== $(date -u +%FT%TZ) vjp $arm $(git -C "$t" rev-parse --short HEAD)"
    d=$o/grad_$arm; mkdir -p "$d"
    (cd "$t" && env $STACK PYTHONPATH=$t timeout 1500 $PY -u perf/bcx_p10_stack2/grad_stack2.py \
        --arm on --out "$d") > "$d.log" 2>&1 || echo "  vjp $arm exited $?"
done
[ "${legs/count/}" = "$legs" ] || {
    echo "=== $(date -u +%FT%TZ) count B"
    d=$o/count_B; mkdir -p "$d"
    (cd "$B" && PYTHONPATH=$B JAX_COMPILATION_CACHE_DIR=$B/perf/bcx_p10_headline/out/xlacache \
        timeout 1500 $PY -u perf/bcx_mainround/count_ln.py --rounds 3 --interleave 1 \
        --trajectories 3 --binder 146 --params /home/ttuser/bcx_e2e/af2_params --out "$d") \
        > "$d.log" 2>&1 || echo "  count exited $?"
}
echo "=== $(date -u +%FT%TZ) acc done"
