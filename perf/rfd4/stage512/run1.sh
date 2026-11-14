#!/bin/bash
# stage512 run1: price fewer, larger stage reads on chip <N>. Tail: main (S1K, 256 x 1 KB reads a unit at P 2) vs
# sbig=S (the same bytes as S-byte contiguous reads rotating over the banks, wrong values) vs dm 2 (no stage reads).
#   bash perf/rfd4/stage512/run1.sh <chip> <outdir>      (from a snapshot tree; books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-stage512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 1500 --now --note "stage512 run1: large stage reads price" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 120 9 || exit 3
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
L="grow=1"
for s in 1024 2048 4096 8192 16384 65536; do L="$L;grow=1,sbig=$s"; done
timeout 1200 $PY perf/rfd4/g512/tail.py --ns 512,384 --legs "$L;grow=1,dm=2" \
  --rounds 3 --reps 20 --out $R/tail.jsonl > $R/tail.log 2>&1
echo "tail rc=$?" >> $R/done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
