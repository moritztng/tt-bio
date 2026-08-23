#!/bin/sh
# Step 3: the 9bk6_164 mechanism check, THREE arms, at OF3 shipped 3/200.
#   A   shipped, no pad, no lever      the arm section 1e measured (-0.00178)
#   Ap  shipped + ragged pad           isolates the pad own effect
#   B   lever + pad                    the arm adoption would ship
# The lever margin is B vs Ap. 164 % 32 = 4, so 28 of 192 key columns entered every
# row softmax unmasked in the historical reading.
set -e
CARD=$1
WT=/home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
cd $WT
export PYTHONPATH=$WT:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:openbind-fused-sdpa-rescore
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"
for ARM in A Ap B; do
  OUT=perf/obfused/anchor/9bk6/$ARM
  [ -f "$OUT/fold.json" ] && { echo "skip $ARM (done)"; continue; }
  case $ARM in
    A)  L="" ;;
    Ap) L="TT_BIO_SDPA_RAGGED_PAD=1" ;;
    B)  L="TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_SDPA_RAGGED_PAD=1" ;;
  esac
  C=$WT/perf/obfused/anchor/9bk6/$ARM.census; rm -rf $C; mkdir -p $C
  echo "########## 9bk6 $ARM [$L] start $(date -u +%H:%M:%S)"
  env $L TT_BIO_SDPA_RAGGED_CENSUS=$C $PY -u perf/obfused/fold_fix_ab_anchor.py \
      --model openfold3 --yaml examples/9bk6.yaml --label "$ARM" \
      --seeds 0,1,2,3,4,0 --dump-distogram --outdir $OUT 2>&1 | grep -viE "$G"
  echo "--- census 9bk6/$ARM ---"; cat $C/ragged_sites_*.json 2>/dev/null || echo NONE
done
echo "########## 9bk6 DONE $(date -u +%H:%M:%S)"
