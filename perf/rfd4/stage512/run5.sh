#!/bin/bash
# stage512 run5: bmmx.py on this tree and on origin/main's (bmm_res shas must match; rot_equal on this tree).
#   bash perf/rfd4/stage512/run5.sh <chip> <outdir> <main tree>
set -u
N=$1; R=$2; MT=$3; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-stage512
mkdir -p $R; R=$(cd $R && pwd)
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
stamp "booked $($BOOK add $OWN $N 1800 --now --note "stage512 run5: bmm_res cross-tree")"
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
cp $WT/perf/rfd4/stage512/bmmx.py $R/bmmx.py
for t in branch main; do
  T=$WT; [ $t = main ] && T=$MT
  (cd $T && PYTHONPATH=$T timeout 1500 $PY $R/bmmx.py --out $R/$t.jsonl > $R/$t.log 2>&1); stamp "$t rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
