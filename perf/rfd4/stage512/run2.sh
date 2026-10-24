#!/bin/bash
# stage512 run2: is the stage bound by its bank (all of a unit on one bank today) or by DRAM rows? Tail: main vs
# sstr=S (the same 1 KB reads stepping S pages a channel instead of St * St, wrong values) vs dm 2.
#   bash perf/rfd4/stage512/run2.sh <chip> <outdir>      (from a snapshot tree; books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-stage512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 1500 --now --note "stage512 run2: page order price" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 120 9 || exit 3
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
L="grow=1"
for s in 1 2 8 9 257 256; do L="$L;grow=1,sstr=$s"; done; L="$L;grow=1,sbig=2048"
timeout 1200 $PY perf/rfd4/g512/tail.py --ns 512,384 --legs "$L;grow=1,dm=2" \
  --rounds 3 --reps 20 --out $R/tail.jsonl > $R/tail.log 2>&1
echo "tail rc=$?" >> $R/done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
