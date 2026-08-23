#!/bin/sh
# OpenFold3's own fold-level evidence for the fused SDPA, which it has never had.
#
# Every OF3 arm ever folded flipped the sites by setting fp32_softmax=False, which takes the
# `else:` branch at tenstorrent.py:3852 -> _tri_att_sdpa(q, k, v, bias, ...) with the bias NOT
# pre-scaled. All four OF3 sites pass scale_pair_bias=False, so _bias_scale is 1.0 while scale is
# not, and that route computes softmax(s*(qk+bias)) instead of softmax(s*qk+bias). That includes
# the arm recorded as `nofp32hifi` in perf/other512/ab_of3_hifi_512.json, which fixed the compute
# config to HiFi4 + fp32_dest_acc and recovered 0.000884 plDDT of a 0.107381 loss -- because the
# loss was never precision.
#
# TT_BIO_TRIATT_FUSED_HIFI=1 takes the OTHER route, tenstorrent.py:3835, which pre-scales the bias
# by scale/_bias_scale before calling the fused kernel. That is the route adoption would take and
# it has never been measured on OF3.
#
# Scored on the distogram, which openfold3_confidence.py:195-197 builds from zij_trunk -- the TRUNK
# pair track, not the confidence pairformer's -- so it is sampler-free in value and blind to the
# confidence-head site, which stays excluded from the flip either way.
#   usage: of3_disto_run.sh <size> <card>
set -e
SIZE=$1
CARD=$2
cd /home/ttuser/.coworker/wt/fused-sdpa-adopt-rf3-of3
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:fused-sdpa-adopt-rf3-of3
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/fused_sdpa/disto_of3/$SIZE
G='fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type'

# the step-count cut has to be proven for openfold3 too, not inherited from rf3
echo "########## of3 proof200 $SIZE start $(date -u +%H:%M:%S)"
$PY -u perf/rf3/fold_fix_ab.py --model openfold3 --fix cdk2x2_$SIZE --label proof200 \
    --seeds 0 --dump-distogram --outdir $OUT/proof200 2>&1 | grep -viE "$G"

for arm in def hifi; do
    if [ "$arm" = hifi ]; then L='TT_BIO_TRIATT_FUSED_HIFI=1'; else L=''; fi
    echo "########## of3 disto $SIZE $arm levers=[$L] start $(date -u +%H:%M:%S)"
    env $L $PY -u perf/rf3/fold_fix_ab.py --model openfold3 --fix cdk2x2_$SIZE --label "$arm" \
        --seeds 0,1,2 --sampling-steps 5 --dump-distogram --outdir $OUT/$arm 2>&1 | grep -viE "$G"
done
echo "########## OF3 DISTO $SIZE ALL DONE $(date -u +%H:%M:%S)"
