#!/bin/bash
# The float64 bar on seeds 1-3 (seed 0 comes from run_chain.sh): whole 4+48 step at n=256, base
# (`stack`) and lever (`bwd`) arms in one process, each arm's logit gradient against float64.
# Correctness only, so it runs beside the timing chain on another card: SEEDS_CARD (default 2).
cd /home/ttuser/.coworker/wt/bcx-bwdplan
C=${SEEDS_CARD:-2}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bcx-bwdplan
export TMPDIR=/dev/shm/bcx-bp-tmp TT_METAL_CACHE=/dev/shm/bcx-bp-cache OMP_NUM_THREADS=8
O=/dev/shm/bcx-bp-out; R=perf/bcx_bwdplan; mkdir -p $O
PY=/home/ttuser/tt-bio-dev/env/bin/python3
for s in ${SEEDS:-1 2 3}; do
  echo "== seed $s $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 2400 taskset -c 0-15 $PY perf/bcx_stack/stack.py whole --arms stack,bwd --n 256 --reps 1 \
    --seed $s --out $O/whole_n256_s$s.json 2>&1 | grep -E "^(stack|bwd)|wrote|Error|Traceback"
  cp $O/whole_n256_s$s.json $R/ 2>/dev/null
done
echo "== done $(date -u +%T)"
