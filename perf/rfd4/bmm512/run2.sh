#!/bin/bash
# bmm512 run2: bmm_res with A_DEPTH 2 / 4 / 8 / 16 row tiles of a in flight (split.py --adepth), two rounds, on chip <N>, AICLK sampled every second.
#   bash perf/rfd4/bmm512/run2.sh <chip> <outdir>      (books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-bmm512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 1500 --now --note "bmm512 run2: A_DEPTH price" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 120 9 || exit 3
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
for d in 2 4 8 16 2 4 8 16; do
  timeout 600 $PY perf/rfd4/bmm512/split.py --n 512,384 --C 220,256 --abl 0,1 --rounds 1 --adepth $d --out $R/split.jsonl >> $R/split.log 2>&1
  echo "adepth $d rc=$?" >> $R/done
done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
