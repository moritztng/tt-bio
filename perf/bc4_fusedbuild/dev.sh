#!/bin/bash
# quick foreground dev run on qb2 chip 2 or 3 under the box flock, bounded by timeout
cd /home/ttuser/.coworker/wt/bc4-bet-fusedbuild
export PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=worker:bc4-bet-fusedbuild
for c in 2 3; do
  TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c flock -n -E 75 /home/ttuser/spd_qb2_chip$c.lock timeout ${T:-240} /home/ttuser/bcx_e2e_venv/bin/python "$@"
  rc=$?; [ $rc -ne 75 ] && exit $rc
done
echo "no chip free"
