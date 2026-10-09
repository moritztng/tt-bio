#!/bin/bash
# BH arm of the accuracy check on qb1 (p150a) at prio 8: takes a card only when its flock has no holder and no
# waiter (fuser empty), then flock -n, so it never queues ahead of an SPD job. One target per acquisition, card
# released between targets; polls every 5 min for 24 h. Run ON qb1.
# usage: bhpoll.sh TREE SET T1[,T2..] [EXTRA run.py args, e.g. "--seeds 101"]
TREE=$1; SET=$2; TARGETS=$3; EXTRA=${4:-}
R=~/spd/tfg-accuracy; PY=/home/ttuser/tt-bio-dev/env/bin/python
LOG=$R/runs/bh-$SET/poll.log; mkdir -p $R/runs/bh-$SET $R/leases; end=$(( $(date +%s) + 86400 ))
for T in ${TARGETS//,/ }; do
  done_t=0
  while [ $done_t = 0 ] && [ $(date +%s) -lt $end ]; do
    for N in 0 1 2 3; do
      L=/home/ttuser/spd_qb1_card$N.lock
      [ -z "$(fuser $L 2>/dev/null)" ] || continue
      exec 9> $L
      flock -n 9 || { exec 9>&-; continue; }
      O=$R/runs/bh-$SET/$T; mkdir -p $O
      echo "$(date -u +%FT%TZ) $T took card $N" >> $LOG
      ( cd $R/trees/$TREE && PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=tfg-accuracy TT_BIO_LEASE_DIR=$R/leases \
          TT_METAL_CACHE=$R/cache-$TREE-$N OMP_NUM_THREADS=2 timeout -s INT 14400 nice -n 10 \
          $PY perf/tfg_acc/run.py --panel $R/data/$SET --target $T --out $O --chip $N --share 8 $EXTRA \
          >> $O/run.log 2>&1 9>&- )
      echo "$(date -u +%FT%TZ) $T card $N end rc=$?" >> $LOG
      exec 9>&-; done_t=1; break
    done
    [ $done_t = 1 ] || sleep 300
  done
done
echo "$(date -u +%FT%TZ) poller exit" >> $LOG
