#!/bin/bash
# tail512 run4: price the candidate levers on the full call at 512 / 384, chip <N>: no sigmoid on the math thread (abl 16),
# no stage reads (dm 2), both, and the gather or LN off beside them.
#   bash perf/rfd4/tail512/run4.sh <chip> <outdir>      (from a snapshot tree; books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tail512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 1800 --now --note "tail512 run4: lever ceilings" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 120 9 || exit 3
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
L="abl=16;dm=2;abl=16,dm=2;abl=1;abl=17;abl=16,dm=1;abl=16,dm=12;abl=16,dm=3;abl=16,dm=4"
timeout 1500 $PY perf/rfd4/tail512/tail.py --ns 512,384 --legs "$L" --rounds 3 --reps 20 --out $R/tail.jsonl \
  > $R/tail.log 2>&1
echo "tail rc=$?" >> $R/done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
