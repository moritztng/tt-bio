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

run_leg() {
  local name="$1" sizes="$2" out="$3" budget="$4"
  for attempt in 1 2 3; do
    echo "leg $name attempt $attempt start $(date -Is)" >> "$STATUS"
    timeout "$budget" "$BL" sizes-recheck-opendde -- \
      $PY -u perf/other512/fold_ab_multi.py --model opendde \
        --sizes "$sizes" --arms on,noqsplit,on --out "perf/oddesizes/$out" \
        > "perf/oddesizes/$name.log" 2>&1
    rc=$?
    echo "leg $name attempt $attempt rc=$rc $(date -Is)" >> "$STATUS"
    if [ $rc -eq 75 ]; then sleep 120; continue; fi
    return $rc
  done
  return 75
}

run_leg leg1 128,256,512 ladder_128_256_512_qb1c2.json 3600
run_leg leg2 768 ladder_768_qb1c2.json 3600
run_leg leg3 1024 ladder_1024_qb1c2.json 5400
run_leg leg4 640 offlattice_640_qb1c2.json 2700
echo "chain done $(date -Is)" >> "$STATUS"
