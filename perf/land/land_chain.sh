#!/bin/bash
# Chain: wait for the fold sweep to finish on card 3, then the W5 cross-model A/B, then the
# gates. Everything here wants the same card, so it is strictly sequential by design.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
cd "$WT" || exit 1
SWEEP="$WT/perf/land/out/sweep.log"
CLOG="$WT/perf/land/out/chain.log"
PY=/usr/bin/python3
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

for _ in $(seq 1 480); do            # up to 4 h
  grep -q "SWEEP COMPLETE" "$SWEEP" && break
  sleep 30
done
grep -q "SWEEP COMPLETE" "$SWEEP" || { echo "sweep never completed, chain aborting" >>"$CLOG"; exit 1; }

echo "=== $(date -u +%H:%M:%S) W5 cross-model A/B" >>"$CLOG"
bash "$WT/perf/land/w5_crossmodel.sh"

gates () {  # armtag treedir fused
  local tag=$1 tree=$2 fused=$3
  export TT_BIO_TRIMUL_OUT_FUSED=$fused
  cd "$WT/$tree" || return 1
  for m in protenix-v2 opendde opendde-abag capacity; do
    local o="$WT/perf/land/out/rg_${tag}_${m}.log"
    [ -s "$o" ] && { echo "SKIP rg $tag $m" >>"$CLOG"; continue; }
    echo "=== $(date -u +%H:%M:%S) release_gate $tag $m" >>"$CLOG"
    PYTHONPATH=$PWD OPENDDE_DOCKQ_PYTHON=$HOME/w6_dockq_py RELEASE_GATE_MSA_DIR=$HOME/w6_gate_msa \
      timeout 3600 "$PY" scripts/release_gate.py --model "$m" >"$o" 2>&1
    echo "=== $(date -u +%H:%M:%S) release_gate $tag $m rc=$?" >>"$CLOG"
  done
  local fj="$WT/perf/land/out/fpg_${tag}.json"
  if [ ! -s "$fj" ]; then
    echo "=== $(date -u +%H:%M:%S) full_parity_gate $tag" >>"$CLOG"
    PYTHONPATH=$PWD OPENDDE_DOCKQ_PYTHON=$HOME/w6_dockq_py \
      timeout 21600 "$PY" scripts/full_parity_gate.py --legacy-rdx --seeds 0,1,2,3,4 --workers qb1:3 \
        --workdir "/home/ttuser/land_fpg_${tag}" --out "$fj" \
        >"$WT/perf/land/out/fpg_${tag}.log" 2>&1
    echo "=== $(date -u +%H:%M:%S) full_parity_gate $tag rc=$?" >>"$CLOG"
  fi
  cd "$WT" || return 1
}

gates L0  arms/L0 0
gates L4F arms/L4 1

echo "=== $(date -u +%H:%M:%S) CHAIN COMPLETE" >>"$CLOG"
