#!/bin/bash
set -u
WT=/home/ttuser/.coworker/wt/sizes-recheck-opendde
cd "$WT"
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:sizes-recheck-opendde
export ESM_ROOT=/home/ttuser/esm
export PYTHONPATH=/home/ttuser/aiand-bio/env/lib/python3.10/site-packages:$WT
PY=/usr/bin/python3.10
STATUS=perf/oddesizes/chain.status
BL=/home/ttuser/.coworker/scripts/benchlock.sh

# timeout bounds ONLY the fold (inside the lock); benchlock's own 5400s governs the wait.
# rc 75 = lock wait timeout, rc 124 = fold exceeded budget: both retry, up to 3 attempts.
run_leg() {
  local name="$1" sizes="$2" out="$3" budget="$4"
  for attempt in 1 2 3; do
    echo "leg $name attempt $attempt start $(date -Is)" >> "$STATUS"
    "$BL" sizes-recheck-opendde -- \
      timeout "$budget" $PY -u perf/other512/fold_ab_multi.py --model opendde \
        --sizes "$sizes" --arms on,noqsplit,on --out "perf/oddesizes/$out" \
        > "perf/oddesizes/$name.log" 2>&1
    rc=$?
    echo "leg $name attempt $attempt rc=$rc $(date -Is)" >> "$STATUS"
    if [ $rc -eq 75 ] || [ $rc -eq 124 ]; then sleep 120; continue; fi
    return $rc
  done
  return 75
}

run_leg leg2 768 ladder_768_qb1c2.json 2700
run_leg leg3 1024 ladder_1024_qb1c2.json 4800
run_leg leg4 640 offlattice_640_qb1c2.json 2400
echo "chain2 done $(date -Is)" >> "$STATUS"
