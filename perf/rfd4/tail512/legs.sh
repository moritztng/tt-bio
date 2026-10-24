#!/bin/bash
# tail512: run tail.py legs at 512 / 384 on chip <N> from a snapshot tree. Holds the chip through book.py (a --now hold
# stops new feed jobs), waits for its flock and for any open device node, samples AICLK each second.
#   bash perf/rfd4/tail512/legs.sh <chip> <outdir> <note> <legs> [rounds]
set -u
N=$1; R=$2; NOTE=$3; L=$4; K=${5:-3}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tail512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 3600 --now --note "$NOTE" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { echo "flock timeout" >> $R/done; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 10; done
echo "start $(date -u +%FT%TZ) tree $(cat .sha 2>/dev/null)" >> $R/done
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
timeout 1800 $PY perf/rfd4/tail512/tail.py --ns 512,384 --legs "$L" --rounds $K --reps 20 --out $R/tail.jsonl \
  > $R/tail.log 2>&1
echo "tail rc=$?" >> $R/done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
