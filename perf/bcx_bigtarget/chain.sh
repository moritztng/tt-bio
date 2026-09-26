#!/bin/bash
# bcx-bigtarget: the remaining rungs, one process per size, in the order that answers first.
cd /home/ttuser/.coworker/wt/bcx-bigtarget || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-bigtarget
PY=/home/ttuser/bcx_e2e_venv/bin/python3
for n in 384 320 288; do
  echo "=== n=$n start $(date -u +%FT%TZ) ==="
  $PY -u perf/bcx_bigtarget/curve.py --n $n --node-blocks evo2,evo1,evo0 --census-every 8 \
      > perf/bcx_bigtarget/n$n.log 2>&1
  echo "=== n=$n rc=$? end $(date -u +%FT%TZ) ==="
done
echo CHAIN-DONE
