#!/bin/bash
# bcx-bigtarget, the ladder's UPPER half on qb1 card 2 (device node 3).
# Waits for the in-flight n=448 (pid 532326) rather than starting a second tenant on the card.
cd /home/ttuser/.coworker/wt/bcx-bigtarget || exit 1
while kill -0 532326 2>/dev/null; do sleep 20; done
echo "=== n=448 finished, card 2 free $(date -u +%FT%TZ) ==="
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-bigtarget
PY=/home/ttuser/bcx_e2e_venv/bin/python3
for n in 512 544; do
  echo "=== n=$n start $(date -u +%FT%TZ) card2/node3 ==="
  $PY -u perf/bcx_bigtarget/curve.py --n $n --census-every 12 > perf/bcx_bigtarget/n$n.log 2>&1
  echo "=== n=$n rc=$? end $(date -u +%FT%TZ) ==="
done
echo UPPER-DONE
