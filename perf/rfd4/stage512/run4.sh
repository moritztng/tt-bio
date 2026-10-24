#!/bin/bash
# stage512 run4: whole designs at 512 through the fair harness (perf/rfd4/fair/run.py, b4 lanes of 1, PD-L1 512),
# base (bmm_res.ROT = False, a copy of this tree) and ROT (this tree) alternated ROUNDS (2) times, 4 seeds each.
#   bash perf/rfd4/stage512/run4.sh <chip> <outdir>      (from a snapshot tree; books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-stage512
mkdir -p $R; R=$(cd $R && pwd)
BT=$WT-base; rm -rf $BT; cp -r $WT $BT
sed -i 's/^ROT = True$/ROT = False/' $BT/tt_bio/bmm_res.py; grep -q '^ROT = False$' $BT/tt_bio/bmm_res.py || exit 4
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-7200} --note "stage512 run4: ROT whole designs 512")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-7200} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat $WT/.sha 2>/dev/null)"
( while sleep 2; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
for k in $(seq ${ROUNDS:-2}); do
  for arm in base rot; do
    T=$WT; [ $arm = base ] && T=$BT
    (cd $T && PYTHONPATH=$T timeout 2400 $PY perf/rfd4/fair/run.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
      --weights $HOME/rfd4-bundle --out $R/$arm$k --part p300c --shared-box --batch 4 --lane-width 1 --batches 4 \
      > $R/$arm$k.log 2>&1)
    stamp "$arm$k rc=$?"
  done
done
kill $CLK
exec 9>&-; $BOOK rm $OWN; rm -rf $BT
stamp ALLDONE
