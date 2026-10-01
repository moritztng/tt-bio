#!/bin/bash
# The 768 axis the orchestrator asked SLOWDOWN to carry: hHSA + 150 (seam axis read back from
# rung.json), looped vs fallback, alternated at the process boundary, qb2 card 1.
set -u
cd /home/ttuser/.coworker/wt/bcw-dbias
out=perf/bcw_dbias/out; log=$out/t768.log
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/$out/xlacache
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcw-dbias
for leg in "t768_on 1" "t768_off 0"; do set -- $leg
  echo "=== $(date -u +%FT%TZ) $1 commit=$(git rev-parse --short HEAD)" >> $log; mkdir -p $out/$1
  TT_BIO_TRIATT_BW_FUSED=$2 timeout 1800 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
    --params /home/ttuser/bcx_e2e/af2_params --out $out/$1 --target hHSA --binder 150 \
    --rounds 3 --trajectories 1 > $out/$1.log 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) $1" >> $log
done
echo "=== T768 DONE $(date -u +%FT%TZ)" >> $log
