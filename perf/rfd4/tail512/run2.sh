#!/bin/bash
# tail512 run2: the compute side of main's tail at 512 / 384 on chip <N> (all DM off: each stage skipped),
# and two bit-identical compute levers: LN stages 4 tiles a DST acquire (lnblk 4), p GEMM subblock width (sbw).
#   bash perf/rfd4/tail512/run1.sh <chip> <outdir>      (from a snapshot tree; books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tail512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 1800 --now --note "tail512 run2: compute split" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 120 9 || exit 3
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
L="dm=15;dm=15,abl=1;dm=15,abl=2;dm=15,abl=8;dm=15,abl=16;dm=15,abl=64;dm=15,abl=91;lnblk=4;sbw=2;sbw=4;lnblk=4,sbw=4;dm=15,lnblk=4;dm=15,sbw=4;dm=15,lnblk=4,sbw=4"
timeout 1500 $PY perf/rfd4/tail512/tail.py --ns 512,384 --legs "$L" --rounds 3 --reps 20 --out $R/tail.jsonl \
  > $R/tail.log 2>&1
echo "tail rc=$?" >> $R/done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
