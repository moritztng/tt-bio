#!/bin/bash
# tail512 run3: the data-movement side of main's tail at 512 / 384 on chip <N> (all compute off, abl 91): every DM
# stream alone and in pairs (dm bits: 1 gather, 2 stage DRAM reads, 4 writer g / z reads, 8 writes).
#   bash perf/rfd4/tail512/run3.sh <chip> <outdir>      (from a snapshot tree; books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tail512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 1800 --now --note "tail512 run3: DM split" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 120 9 || exit 3
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
L="abl=91,dm=1;abl=91,dm=2;abl=91,dm=4;abl=91,dm=8;abl=91,dm=5;abl=91,dm=6;abl=91,dm=9;abl=91,dm=10;abl=91,dm=13;abl=91,dm=14;abl=91,dm=11;abl=91,dm=7;abl=91,dm=3;abl=91,dm=12;abl=91;abl=91,dm=15"
timeout 1500 $PY perf/rfd4/tail512/tail.py --ns 512,384 --legs "$L" --rounds 3 --reps 20 --out $R/tail.jsonl \
  > $R/tail.log 2>&1
echo "tail rc=$?" >> $R/done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
