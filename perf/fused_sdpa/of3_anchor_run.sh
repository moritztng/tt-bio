#!/bin/sh
# One OF3 anchor, one arm, five scored seeds plus an A/A repeat, on one card.
#
# The arm is TT_BIO_TRIATT_FUSED_HIFI=1, which is the route adoption would ship: it takes
# tenstorrent.py:3980 and pre-scales the pair bias by scale/_bias_scale before the fused kernel.
# Every historical OF3 fused arm instead set fp32_softmax=False and landed on the raw-bias route,
# so none of them measured this kernel (state/fused-sdpa-adopt.md 1b).
#
# Folds at OF3's shipped 3 recycles / 200 sampling steps -- fold_fix_ab.py asserts that pair, so
# the coordinates are production coordinates and the structural corroborators come free.
#   usage: of3_anchor_run.sh <fixture> <arm def|hifi> <card>
set -e
FIX=$1
ARM=$2
CARD=$3
cd /home/ttuser/.coworker/wt/fused-sdpa-adopt-of3-p2
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:fused-sdpa-adopt-of3-p2
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"
if [ "$ARM" = hifi ]; then L="TT_BIO_TRIATT_FUSED_HIFI=1"; else L=""; fi
echo "########## $FIX $ARM card=$CARD levers=[$L] start $(date -u +%H:%M:%S)"
env $L $PY -u perf/rf3/fold_fix_ab.py --model openfold3 --fix "$FIX" --label "$ARM" \
    --seeds 0,1,2,3,4,0 --dump-distogram \
    --outdir perf/fused_sdpa/anchor/"$FIX"/"$ARM" 2>&1 | grep -viE "$G"
echo "########## $FIX $ARM DONE $(date -u +%H:%M:%S)"
