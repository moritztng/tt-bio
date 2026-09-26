#!/bin/bash
# bcx-bigtarget, the ladder's LOWER half, fanned onto qb1 card 3 (device node 0).
#
# Offered by bcx-orchestrator 2026-09-26 07:1xZ after qb1_s12 exited rc=0. Checked live before
# taking it: `fuser /dev/tenstorrent/0` empty, no cardblock-qb1-3, previous holder pid 60764 gone.
# The grant is widened on this command only, per the card-fanout rule. Cards 0 and 1 carry the
# acceptance denominator and are not touched.
#
# The ladder's points are independent, so the lower half runs here while card 2 walks the upper.
cd /home/ttuser/.coworker/wt/bcx-bigtarget || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=2,3 TT_BIO_LEASE_HOLDER=worker:bcx-bigtarget
PY=/home/ttuser/bcx_e2e_venv/bin/python3
for n in 288 320 256; do
  echo "=== n=$n start $(date -u +%FT%TZ) card3/node0 ==="
  $PY -u perf/bcx_bigtarget/curve.py --n $n --census-every 12 > perf/bcx_bigtarget/n$n.log 2>&1
  echo "=== n=$n rc=$? end $(date -u +%FT%TZ) ==="
done
echo LOWER-DONE
