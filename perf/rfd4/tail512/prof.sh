#!/bin/bash
# tail512 prof: the T2CM's DEVICE KERNEL DURATION inside the real trunk at 512 (elt_prof.sh, Tracy build, main code:
# every bench knob off), to compare against tail.py's 1.78 ms and its slowed 2.2-2.5 ms state (run8, run11).
#   bash perf/rfd4/tail512/prof.sh <chip> <outdir> [n]
set -u
N=$1; R=$2; NN=${3:-512}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tail512
mkdir -p $R; R=$(cd $R && pwd)
$BOOK add $OWN $N 3600 --now --note "tail512 prof: in-trunk T2CM at $NN" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { echo "flock timeout" >> $R/done; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 10; done
echo "start $(date -u +%FT%TZ) tree $(cat $WT/.sha 2>/dev/null)" >> $R/done
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
TT_BIO_LEASE_HOLDER=worker:$OWN bash $WT/perf/rfd4/e2e/elt_prof.sh $R $N --n $NN --cycles 2 > $R/prof.log 2>&1
echo "prof rc=$?" >> $R/done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
