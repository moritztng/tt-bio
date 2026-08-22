#!/bin/sh
# P20 item 3: score the two TriangleAttention bias forms on RF3's own distogram.
#   arm `none` = main's shipped form, both biases via a separate ttnn.add_
#   arm `g`    = L-G, linear_g.bias inside its minimal_matmul
#   arm `o`    = L-O, linear_o.bias inside the output projection (this branch's default)
# usage: p20_bias_disto.sh <size> <card> <arm>...
set -e
SIZE=$1; CARD=$2; shift 2
cd /home/ttuser/.coworker/wt/pxdesign-af2ig-port-p20
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:pxdesign-af2ig-port-p20
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/fused_sdpa/disto_bias/$SIZE
G='fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type'
for arm in "$@"; do
    echo "########## disto_bias $SIZE arm=$arm start $(date -u +%H:%M:%S)"
    env TT_BIO_PAIR_BIAS_IN_MATMUL=$arm $PY -u perf/rf3/fold_fix_ab.py \
        --fix cdk2x2_$SIZE --label "$arm" --seeds 0,1,2 --sampling-steps 5 \
        --dump-distogram --outdir $OUT/$arm 2>&1 | grep -viE "$G"
done
echo "########## DISTO_BIAS $SIZE DONE $(date -u +%H:%M:%S)"
