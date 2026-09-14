#!/bin/bash
# The quiet-box retake of the fold A/B, on card 0, held behind benchlock.
#
# Pass 5 measured on a box carrying three 44-leg gates and the bracketed A/A floor came out at
# 1.07422x against a lever worth about 1.01x -- unreadable, not slow. This waits for the box
# instead of arguing with it: benchlock only proceeds below load 2.0, and the wait is set long
# enough to outlast every gate on the box right now. Exit 75 means the wait expired and NO
# number may be read out of the run.
#
# It shares benchlock with the card-3 A/B the pass-3 chain queues behind its gate, so the two
# cannot run at once and whichever reaches a quiet box first is the reading.
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
cd $W
export PYTHONPATH=$W
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=$W/perf/b2z2_zinit_ship
L=$O/pass5_quiet_ab.log
echo "[$(date -Is)] waiting for benchlock on a quiet box" >> $L
( while :; do echo "$(date -Is) $(cut -d" " -f1-3 /proc/loadavg)"; sleep 30; done ) > $O/quiet_load_c0.txt &
LOG=$!
BENCHLOCK_WAIT_S=39600 BENCHLOCK_LOAD_WAIT_S=36000 \
  /home/ttuser/.coworker/scripts/benchlock.sh b2z2-zinit-ship -- \
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship \
  $PY -u perf/b2z2_cond/fold_cond.py --flag TT_BIO_DEVICE_ZINIT --reader _device_zinit \
  --sizes 512 --timing-reps 10 --timing-arms base,on,base \
  --out $O/fold_ab512_quiet_c0.json --cifdir $O/ab_cifs_quiet_c0 >> $L 2>&1
rc=$?
kill $LOG 2>/dev/null
echo "[$(date -Is)] fold A/B exited $rc (75 = benchlock timed out, read no number)" >> $L
[ $rc -eq 0 ] && $PY $O/pair_ab.py $O/fold_ab512_quiet_c0.json > $O/pair_ab512_quiet_c0.json 2>&1
echo "[$(date -Is)] QUIETABDONE" >> $L
