#!/bin/bash
# Trunk attribution, 256 -> 512 -> 1024 on qb1 card 3 (booked fleet-mol1-bh-trunk), under the card flock.
W=/home/ttuser/.coworker/wt/mol1-bh-trunk; PY=/home/ttuser/tt-bio-dev/env/bin/python; O=/home/ttuser/mol1-bhtrunk/prof
mkdir -p $O; cd /home/ttuser/mol1-fair/inputs
exec 9>/home/ttuser/spd_qb1_card3.lock; flock -w 600 9 || { echo "no flock" >> $O/run.log; exit 1; }
for s in 256 512 1024; do
  echo "$(date -u +%FT%TZ) start $s" >> $O/run.log
  TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:mol1-bh-trunk PYTHONPATH=$W \
    timeout 5400 $PY $W/perf/mol1_bhtrunk/prof.py --input mol1-$s.json --out $O/s$s --warm $([ $s = 1024 ] && echo 1 || echo 2) >> $O/run.log 2>&1
  echo "$(date -u +%FT%TZ) end $s rc=$?" >> $O/run.log
done
touch $O/DONE
