#!/bin/bash
# Step 2: the prize on OB0's own bar cells. ABBA per rung, one benchlock per rung.
# Box is co-tenanted (rf3-4x + pxdesign live); short load wait, loadavg recorded per arm.
WT=/home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
PY=/home/ttuser/tt-bio-dev/env/bin/python3
LOCK=$HOME/.coworker/scripts/benchlock.sh
cd $WT
mkdir -p perf/obfused/prize
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
run_rung(){
  AA=$1; REP=$2
  for ARM in A B B A; do
    N=$(ls perf/obfused/prize/${AA}_${ARM}_*.json 2>/dev/null | wc -l)
    OUT=perf/obfused/prize/${AA}_${ARM}_${N}.json
    if [ "$ARM" = B ]; then L="TT_BIO_TRIATT_FUSED_HIFI=1"; else L="TT_BIO_TRIATT_FUSED_HIFI=0"; fi
    echo "##### rung $AA arm $ARM rep $REP start $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
        TT_BIO_LEASE_HOLDER=worker:openbind-fused-sdpa-rescore \
        BENCHLOCK_LOAD_WAIT_S=90 BENCHLOCK_WAIT_S=420 \
      $LOCK openbind-fused-${AA}aa -- env $L PYTHONPATH=$WT $PY -u \
        perf/openbind/tt_ob_run.py --model openfold3 \
        --input perf/openbind/inputs/ob_apo_${AA}.tt.yaml --repeat $REP --label "$ARM" \
        --out $OUT 2>&1 | grep -viE "fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type" | tail -14
    echo "##### rung $AA arm $ARM done $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  done
}
run_rung 512 2
echo "=== 512 RUNG DONE $(date -u +%H:%M:%S) ==="
run_rung 768 2
echo "=== 768 RUNG DONE $(date -u +%H:%M:%S) ==="
