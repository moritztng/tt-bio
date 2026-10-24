#!/bin/bash
# tri512 run9: X8 price on chip <N>. Tail: S1K main vs dm 48 (half stage + gather bytes) vs dm 112 (+ the bfp8
# gather's exponent byte copies) vs dm 2 (no stage reads), 3 rounds. Contraction: bmm_res bf16 vs out8 vs no write.
#   bash perf/rfd4/tri512/run9.sh <chip> <outdir>      (from a snapshot tree; books and locks the chip itself)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tri512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
$BOOK add $OWN $N 1200 --now --note "tri512 run9: X8 price" > /dev/null
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 120 9 || exit 3
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
timeout 700 $PY perf/rfd4/g512/tail.py --ns 512,384 --legs "grow=1;grow=1,dm=48;grow=1,dm=112;grow=1,dm=2" \
  --rounds 3 --reps 20 --out $R/tail.jsonl > $R/tail.log 2>&1
echo "tail rc=$?" >> $R/done
for tr in none a; do
  timeout 500 $PY perf/spd/af2_trimul_bmm_sweep.py --out $R/bmm-$tr.json --sizes 512 --channels 256 --dtype bfp8 \
    --fid HiFi2 --res --abl 2 --transpose $tr --reps 5 --inner 5 --no-reuse2 > $R/bmm-$tr.log 2>&1
  echo "bmm-$tr rc=$?" >> $R/done
done
kill $CLK
exec 9>&-; $BOOK rm $OWN
echo ALLDONE >> $R/done
