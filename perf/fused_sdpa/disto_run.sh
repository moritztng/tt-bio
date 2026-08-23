#!/bin/sh
# The sampler-free pair-track metric on the cdk2x2_* fixture family, one size, both arms.
#
# RF3s distogram is linear(z + z.T) computed at rf3/model.py:324, BEFORE sampler.sample at :338.
# It is a direct readout of the trunk pair representation, which is where triangle attention
# lives, and it carries no sampler noise -- so it separates the arms on a fixture whose global
# CA RMSD only reports which basin the sampler drew.
#
# Sampling steps cut 50 -> 5. The distogram is computed before the sampler runs, so the cut is
# exact rather than an approximation, and it is already proven byte-identical on THIS fixture
# family (state/fused-sdpa-adopt.md, proof50). Set PROOF50=1 to re-prove it on a new family.
#
# No benchlock: the distogram is scored on values, host contention cannot move one, and the
# fold_s numbers this prints are NOT perf numbers.
#
#   usage:              disto_run.sh <size> <card>
#   third arm:  ARMS="hifipad" disto_run.sh 298 3
#
# ARMS selects which arms run, so a re-fold does not redo the two that are already on disk:
#   def      shipped route
#   hifi     TT_BIO_TRIATT_FUSED_HIFI=1
#   hifipad  the same plus TT_BIO_SDPA_RAGGED_PAD=1, which needs the ragged fix merged in
set -e
SIZE=$1
CARD=$2
WT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$WT"
export PYTHONPATH=$WT:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:$(basename "$WT")}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/fused_sdpa/disto/$SIZE
ARMS=${ARMS:-"def hifi"}
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"

if [ -n "$PROOF50" ]; then
    echo "########## proof50 start $(date -u +%H:%M:%S)"
    $PY -u perf/rf3/fold_fix_ab.py --fix cdk2x2_$SIZE --label proof50 \
        --seeds 0 --dump-distogram --outdir $OUT/proof50 2>&1 | grep -viE "$G"
fi

for arm in $ARMS; do
    case $arm in
        def)     L="" ;;
        hifi)    L="TT_BIO_TRIATT_FUSED_HIFI=1" ;;
        hifipad) L="TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_SDPA_RAGGED_PAD=1" ;;
        *)       echo "unknown arm $arm" >&2; exit 2 ;;
    esac
    echo "########## disto $SIZE $arm levers=[$L] start $(date -u +%H:%M:%S)"
    env $L $PY -u perf/rf3/fold_fix_ab.py --fix cdk2x2_$SIZE --label "$arm" \
        --seeds 0,1,2 --sampling-steps 5 --dump-distogram --outdir $OUT/$arm 2>&1 | grep -viE "$G"
done
echo "########## DISTO $SIZE ALL DONE $(date -u +%H:%M:%S)"
