#!/bin/bash
# bcx-bigtarget, pass 2. The ceiling moved up, so the ladder goes up first.
#
# No --node-blocks: at n=352 the instantaneous peak was 11.409 GB against a 10.775 GB resident
# peak, so the block-boundary read is within 5.9 % of the true high-water and it costs a third
# of the wall clock. The ceiling rung gets node sampling once it is known.
cd /home/ttuser/.coworker/wt/bcx-bigtarget || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-bigtarget
PY=/home/ttuser/bcx_e2e_venv/bin/python3
for n in 448 288 320 512; do
  echo "=== n=$n start $(date -u +%FT%TZ) ==="
  $PY -u perf/bcx_bigtarget/curve.py --n $n --census-every 12 > perf/bcx_bigtarget/n$n.log 2>&1
  echo "=== n=$n rc=$? end $(date -u +%FT%TZ) ==="
done
echo CHAIN2-DONE
