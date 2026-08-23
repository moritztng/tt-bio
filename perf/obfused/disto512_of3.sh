#!/bin/sh
# Step 4: the powered 512 rung for OPENFOLD3 (not rf3). Own worktree, own lease holder.
# Derived from perf/fused_sdpa/disto_multi_run.sh, which hardcodes another worker
# worktree and omits --model. No benchlock: nothing here is a perf number.
#   usage: disto512_of3.sh <card> <seeds> <target> [...]
set -e
CARD=$1; SEEDS=$2; shift 2
WT=/home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
cd $WT
export PYTHONPATH=$WT:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:openbind-fused-sdpa-rescore
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"
for FIX in "$@"; do
  for ARM in def hifi; do
    OUT=perf/obfused/disto512/$FIX/$ARM
    if [ -f "$OUT/fold.json" ]; then echo "skip $FIX/$ARM (done)"; continue; fi
    if [ "$ARM" = hifi ]; then L="TT_BIO_TRIATT_FUSED_HIFI=1"; else L=""; fi
    C=$WT/perf/obfused/disto512/$FIX/$ARM.census; rm -rf $C; mkdir -p $C
    echo "########## 512 $FIX $ARM [$L] start $(date -u +%H:%M:%S)"
    env $L TT_BIO_SDPA_RAGGED_CENSUS=$C $PY -u perf/rf3/fold_fix_ab.py \
        --model openfold3 --fix $FIX --label "$ARM" \
        --fixdir perf/fused_sdpa/targets --seeds $SEEDS --sampling-steps 5 \
        --dump-distogram --outdir $OUT 2>&1 | grep -viE "$G"
    echo "--- census $FIX/$ARM ---"; cat $C/ragged_sites_*.json 2>/dev/null || echo NONE
  done
done
echo "########## RUNG 512 DONE $(date -u +%H:%M:%S)"
