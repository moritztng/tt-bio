#!/bin/bash
# Pass 5: the Blackhole fold A/B for TT_BIO_DEVICE_ZINIT, taken on card 0 (this launch grant)
# while the row own 44-leg gate holds card 3 and two sibling gates hold cards 1 and 2.
#
# NOT benchlocked, deliberately: three 44-leg gates have held this box since 23:00Z and
# benchlock only proceeds below load 2.0, so waiting for it is waiting for tomorrow. The run
# therefore logs loadavg beside every fold and the A/A floor is read out of the same session.
# A host-work lever measured under host contention reads HIGH, so whatever comes out is an
# UPPER BOUND on the quiet-box ratio, and the chain quiet reading still supersedes it.
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
cd $W
export PYTHONPATH=$W
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=$W/perf/b2z2_zinit_ship
( while :; do echo "$(date -Is) $(cut -d" " -f1-3 /proc/loadavg)"; sleep 15; done ) > $O/pass5_load_c0.txt &
LOG=$!
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship \
  $PY -u perf/b2z2_cond/fold_cond.py --flag TT_BIO_DEVICE_ZINIT --reader _device_zinit \
  --sizes 512 --timing-reps 10 --timing-arms base,on,base \
  --out $O/fold_ab512_qb2_c0.json --cifdir $O/ab_cifs_c0 > $O/pass5_ab_c0.log 2>&1
rc=$?
kill $LOG 2>/dev/null
echo "fold A/B exited $rc" >> $O/pass5_ab_c0.log
