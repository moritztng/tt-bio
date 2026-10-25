#!/bin/bash
# g512 run1 (pricing, main + bench knobs only): the T2CM tail at 512 / 384 as the trunk runs it, with data-movement
# and compute stage skips and the row-grouped gather (tail.py); census v11 at 512 b1 and 384 b1 on this tree.
# Books its chip, waits for the window, takes the chip flock.
#   bash perf/rfd4/g512/run1.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-512g
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-2700} --note "g512 run1: tail split + census 512/384")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-2700} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat TREE 2>/dev/null)"
TL="dm=1;dm=2;dm=4;dm=8;dm=12;dm=3;abl=27;abl=27,dm=1;abl=27,dm=12;abl=27,dm=15;grow=1"
for leg in ${3:-tail census512 census384}; do
  case $leg in
    tail) timeout 1500 $PY perf/rfd4/g512/tail.py --ns 512,384 --legs "$TL" --out $R/tail.jsonl ;;
    census*) n=${leg#census}
      bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n \
        && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-$n-v11.json \
        && python3 perf/rfd4/land10/top_ops.py $R/$leg $R/top-ops-$n.json > $R/top-ops-$n.txt ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
