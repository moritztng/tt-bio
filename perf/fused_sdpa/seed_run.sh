#!/bin/sh
# The verdict run: RF3 cdk2x2_298, shipped vs TT_BIO_TRIATT_FUSED_HIFI=1, five diffusion seeds
# each, scored against the 1HCL crystal (not against each other, and not against the incumbent).
#
# Seeds 0,1,2,3,4,0 -- seed 0 twice, so the last warm fold is an A/A control against the first in
# the SAME process. Byte-identical CIFs there mean the arm is deterministic and nothing drifted
# over the run; a difference between seed 0 and seed 1 means the seed is actually live.
#
# No benchlock: these folds are scored on coordinates, so host contention cannot move the answer,
# and holding a host-scoped lock for ~25 min starves every other worker on qb2. The fold_s numbers
# this prints are NOT perf numbers.
cd /home/ttuser/.coworker/wt/fused-sdpa-adopt-rf3-of3 || exit 1
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:fused-sdpa-adopt-rf3-of3
PY=/home/ttuser/tt-bio-dev/env/bin/python3
G='fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type'
for arm in def hifi; do
    if [ "$arm" = hifi ]; then L='TT_BIO_TRIATT_FUSED_HIFI=1'; else L=''; fi
    echo "########## seedrun $arm levers=[$L] start $(date -u +%H:%M:%S)"
    env $L $PY -u perf/rf3/fold_fix_ab.py --fix cdk2x2_298 --label "$arm" \
        --seeds 0,1,2,3,4,0 --outdir perf/fused_sdpa/seeds/$arm 2>&1 | grep -viE "$G"
    echo "########## seedrun $arm done $(date -u +%H:%M:%S)"
done
echo '########## SEEDRUN ALL DONE'
