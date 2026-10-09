#!/bin/bash
# BH arm of the accuracy check (D10): ONE plain flock waiter on qb1 card0, queued behind whoever already waits, so it
# never jumps an SPD job. Holds the card at most CAP s in total (default 45 min), one run.py process per target, card
# released at the end. run.py records the AICLK sampled during every fold. Run ON qb1.
# usage: bhflock.sh TREE SET T1[,T2..] [EXTRA run.py args] [CAP]
TREE=$1; SET=$2; TARGETS=$3; EXTRA=${4:-}; CAP=${5:-2700}; N=0
R=~/spd/tfg-accuracy; PY=/home/ttuser/tt-bio-dev/env/bin/python; O=$R/runs/bh-$SET; mkdir -p $O $R/leases
exec 9> /home/ttuser/spd_qb1_card$N.lock
echo "$(date -u +%FT%TZ) waiting card $N" >> $O/flock.log
flock 9
start=$(date +%s); echo "$(date -u +%FT%TZ) took card $N" >> $O/flock.log
for T in ${TARGETS//,/ }; do
  left=$(( start + CAP - $(date +%s) )); [ $left -gt 120 ] || break
  mkdir -p $O/$T
  ( cd $R/trees/$TREE && PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=tfg-accuracy TT_BIO_LEASE_DIR=$R/leases \
      TT_METAL_CACHE=$R/cache-$TREE-$N OMP_NUM_THREADS=2 timeout -s INT $left nice -n 10 \
      $PY perf/tfg_acc/run.py --panel $R/data/$SET --target $T --out $O/$T --chip $N --share 8 $EXTRA \
      >> $O/$T/run.log 2>&1 9>&- )
  echo "$(date -u +%FT%TZ) $T end rc=$?" >> $O/flock.log
done
echo "$(date -u +%FT%TZ) released card $N" >> $O/flock.log
