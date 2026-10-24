#!/bin/bash
# tail512 run5: writer depth (wd: g / z blocks the reads run ahead) at 512 / 384 on chip <N>, with and without the sigmoid (abl 16),
# and the writer's reads and writes apart beside compute without the sigmoid.
#   bash perf/rfd4/tail512/run5.sh <chip> <outdir>      (from a snapshot tree; books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tail512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 1800 --now --note "tail512 run5: writer depth" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 120 9 || exit 3
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
L="wd=3;wd=4;abl=16;abl=16,wd=3;abl=16,wd=4;abl=16,dm=8;abl=16,dm=11;abl=16,dm=7;abl=16,dm=3,wd=4"
timeout 1500 $PY perf/rfd4/tail512/tail.py --ns 512,384 --legs "$L" --rounds 3 --reps 20 --out $R/tail.jsonl \
  > $R/tail.log 2>&1
echo "tail rc=$?" >> $R/done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
