#!/usr/bin/env bash
# Re-baseline both arms of PAIR_FFN_L1_FC1 on current main, four sizes plus the OpenDDE
# consumer-census leg. One process per size, arms round-robin, everything under benchlock.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-3p4x-close
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:esmfold2-3p4x-close PYTHONPATH="$WT"

run() {  # run <label> <extra fold_ab args...>
  local label="$1"; shift
  echo "=== $label start $(date -Is) ==="
  /home/ttuser/.coworker/scripts/benchlock.sh esmfold2-3p4x-close -- \
    "$PY" -u perf/esm3p4land/fold_ab.py "$@"
  echo "=== $label rc=$? end $(date -Is) ==="
}

run esm512  --model esmfold2 --size 512  --rounds 3 --arms base,l2 --out perf/esm3p4close/fold_ab_512_c1.json
run odde512 --model opendde  --size 512  --rounds 1 --arms base,l2 --out perf/esm3p4close/fold_ab_odde512_c1.json
run esm298  --model esmfold2 --size 298  --rounds 3 --arms base,l2 --out perf/esm3p4close/fold_ab_298_c1.json
run esm768  --model esmfold2 --size 768  --rounds 3 --arms base,l2 --out perf/esm3p4close/fold_ab_768_c1.json
run esm1024 --model esmfold2 --size 1024 --rounds 3 --arms base,l2 --out perf/esm3p4close/fold_ab_1024_c1.json
echo "=== SWEEP DONE $(date -Is) ==="
