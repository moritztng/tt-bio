#!/bin/bash
# Pass 2 of the rescore: the three cells section 39.6 named as owed.
#   phase 1, benchlocked: ob_apo_768 arm B twice (B/B control) + a second arm A (A/A control)
#   phase 2, no lock:     hema_512's missing lever-off arm, then the 9bk6_164 three-arm anchor
# Phase 2 is fidelity only, so it needs no quiet box; benchlock's own foreign-fold detection
# keeps it from contaminating anyone else's timed run.
WT=/home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
PY=/home/ttuser/tt-bio-dev/env/bin/python3
LOCK=$HOME/.coworker/scripts/benchlock.sh
cd $WT
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt ESM_ROOT=/home/ttuser/esm
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"

for SPEC in B:0 B:1 A:1; do
  ARM=${SPEC%%:*}; N=${SPEC##*:}
  OUT=perf/obfused/prize/768_${ARM}_${N}.json
  [ -f "$OUT" ] && { echo "skip 768 $ARM $N (done)"; continue; }
  if [ "$ARM" = B ]; then L="TT_BIO_TRIATT_FUSED_HIFI=1"; else L="TT_BIO_TRIATT_FUSED_HIFI=0"; fi
  echo "##### 768 arm $ARM ($N) start $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:openbind-fused-sdpa-rescore \
      BENCHLOCK_WAIT_S=2400 BENCHLOCK_LOAD_WAIT_S=120 \
    $LOCK openbind-fused-768aa -- env $L PYTHONPATH=$WT $PY -u \
      perf/openbind/tt_ob_run.py --model openfold3 \
      --input perf/openbind/inputs/ob_apo_768.tt.yaml --repeat 2 --label "$ARM" --out $OUT \
      2>&1 | grep -viE "$G" | tail -12
  echo "##### 768 arm $ARM ($N) done $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "=== 768 RUNG DONE $(date -u +%H:%M:%S) ==="

bash perf/obfused/disto512_of3.sh 1 0,1,2 hema_512
echo "=== HEMA DONE $(date -u +%H:%M:%S) ==="

bash perf/obfused/anchor9bk6.sh 1
echo "=== ANCHOR DONE $(date -u +%H:%M:%S) ==="
