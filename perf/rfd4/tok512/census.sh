#!/bin/bash
# rfd4-perf-tok512 census on one qb2 chip: device profiler per-op tables at 512 b4, 512 b1 and 224 b20 on this tree.
# Books the chip, waits for the window, takes the chip flock, writes top_ops tables per leg.
#   bash perf/rfd4/tok512/census.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tok512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-5400} --note "RFD4 tok512 census")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-5400} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
for leg in ${3:-c512b4 c224b20 c512b1}; do
  n=${leg:1:3}; b=${leg#*b}
  { bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n --batch $b \
      && python3 perf/rfd4/land10/top_ops.py $R/$leg $R/top-ops-$leg.json > $R/top-ops-$leg.txt; } > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
