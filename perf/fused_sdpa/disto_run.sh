#!/bin/sh
# Step 3 of state/fused-sdpa-adopt_PLAN2.md: the sampler-free pair-track metric.
#
# RF3's distogram is linear(z + z.T) computed at rf3/model.py:324, BEFORE sampler.sample at :338.
# It is a direct readout of the trunk pair representation, which is where triangle attention
# lives, and it carries no sampler noise -- so it separates the arms on a fixture whose global
# CA RMSD only reports which basin the sampler drew.
#
# Sampling steps cut 50 -> 5. The distogram is computed before the sampler runs, so this is exact
# rather than an approximation, and the first run below proves it: def/seed0 at 50 steps must give
# a byte-identical distogram to def/seed0 at 5 steps.
#
# No benchlock: the distogram is scored on values, host contention cannot move one, and the
# fold_s numbers this prints are NOT perf numbers.
#   usage: disto_run.sh <size> <card>
set -e
SIZE=$1
CARD=$2
cd /home/ttuser/.coworker/wt/fused-sdpa-adopt-rf3-of3
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:fused-sdpa-adopt-rf3-of3
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/fused_sdpa/disto/$SIZE
G='fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type'

if [ "$SIZE" = 298 ]; then
    echo "########## proof50 start $(date -u +%H:%M:%S)"
    $PY -u perf/rf3/fold_fix_ab.py --fix cdk2x2_$SIZE --label proof50 \
        --seeds 0 --dump-distogram --outdir $OUT/proof50 2>&1 | grep -viE "$G"
fi

for arm in def hifi; do
    if [ "$arm" = hifi ]; then L='TT_BIO_TRIATT_FUSED_HIFI=1'; else L=''; fi
    echo "########## disto $SIZE $arm levers=[$L] start $(date -u +%H:%M:%S)"
    env $L $PY -u perf/rf3/fold_fix_ab.py --fix cdk2x2_$SIZE --label "$arm" \
        --seeds 0,1,2 --sampling-steps 5 --dump-distogram --outdir $OUT/$arm 2>&1 | grep -viE "$G"
done
echo "########## DISTO $SIZE ALL DONE $(date -u +%H:%M:%S)"
