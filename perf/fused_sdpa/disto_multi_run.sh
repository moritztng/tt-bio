#!/bin/sh
# One rung of the multi-target distogram A/B. Both arms, all seeds, one card, one target at a time.
#
#   usage: disto_multi_run.sh <rung> <card> <seeds> <target> [<target> ...]
#   e.g.   sh perf/fused_sdpa/disto_multi_run.sh 512 1 0,1,2 chox_499 amyl_496 hutH_507
#
# Sampling steps are cut 50 -> 5. The distogram is `linear(z + z.T)` at rf3/model.py:324, computed
# BEFORE `sampler.sample` at :338, so the cut is exact rather than an approximation -- proven
# byte-identical once already (state/fused-sdpa-adopt.md, proof50). Re-prove it on the first target
# of a new fixture family rather than inheriting it: `--seeds 0` with and without the cut.
#
# No benchlock. Nothing here is a perf number; the fold_s values are run-integrity only, and a
# host-scoped lock held for an hour starves every other worker on the box.
set -e
RUNG=$1; CARD=$2; SEEDS=$3; shift 3
WT=/home/ttuser/.coworker/wt/rf3-fused-sdpa-fixture-power
cd $WT
export PYTHONPATH=$WT:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:rf3-fused-sdpa-fixture-power
PY=/home/ttuser/tt-bio-dev/env/bin/python3
G='fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type'

for FIX in "$@"; do
  for ARM in def hifi; do
    OUT=perf/fused_sdpa/disto_multi/$RUNG/$FIX/$ARM
    if [ -f "$OUT/fold.json" ]; then echo "skip $FIX/$ARM (done)"; continue; fi
    if [ "$ARM" = hifi ]; then L='TT_BIO_TRIATT_FUSED_HIFI=1'; else L=''; fi
    echo "########## $RUNG $FIX $ARM [$L] start $(date -u +%H:%M:%S)"
    env $L $PY -u perf/rf3/fold_fix_ab.py --fix $FIX --label "$ARM" \
        --fixdir perf/fused_sdpa/targets --seeds $SEEDS --sampling-steps 5 \
        --dump-distogram --outdir $OUT 2>&1 | grep -viE "$G"
  done
done
echo "########## RUNG $RUNG DONE $(date -u +%H:%M:%S)"
