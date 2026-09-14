#!/usr/bin/env bash
# B3 ship verification on qb2 card 0 (Blackhole p300c), current main head.
# One process per leg, arms round-robin inside the process, every timed leg under benchlock.
set -u
WT=/home/ttuser/.coworker/wt/ttx-b3-pairffn-fc1-l1-ship
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
       TT_BIO_LEASE_HOLDER=worker:ttx-b3-pairffn-fc1-l1-ship PYTHONPATH="$WT"

run() {  # run <label> <fold_ab args...>
  local label="$1"; shift
  echo "=== $label start $(date -Is) ==="
  /home/ttuser/.coworker/scripts/benchlock.sh ttx-b3-pairffn-fc1-l1-ship -- \
    "$PY" -u perf/esm3p4land/fold_ab.py "$@"
  echo "=== $label rc=$? end $(date -Is) ==="
}

run esm512 --model esmfold2 --size 512 --rounds 3 --arms base,l2 \
  --out perf/ttx_b3/fold_ab_esm512_c0.json
run b2_512 --model boltz2 --size 512 --rounds 1 --arms base,l2 \
  --out perf/ttx_b3/fold_ab_b2_512_c0.json
run esm768 --model esmfold2 --size 768 --rounds 2 --arms base,l2 \
  --out perf/ttx_b3/fold_ab_esm768_c0.json
run px2_512 --model protenix-v2 --size 512 --rounds 1 --arms base,l2 \
  --out perf/ttx_b3/fold_ab_px2_512_c0.json
echo "=== B3 RUN DONE $(date -Is) ==="
