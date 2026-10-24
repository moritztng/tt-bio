#!/bin/bash
# rfd4-perf-p512 census: call_prof (per-call head/tail device time) on abba.sh's trees A and P, after the ABBA run.
#   bash perf/rfd4/p512/census.sh <chip> <abba outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd); OWN=rfd4-perf-p512; MATE=$((N ^ 1))
BOOK="python3 $HOME/spd/feed/book.py"
until grep -q ALLDONE $R/done 2>/dev/null; do sleep 60; done
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "census booked $($BOOK add $OWN $N,$MATE 3600 --note "p512 call census")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N,$MATE 3600 > /dev/null; sleep 60; done
exec 8>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 8 || { stamp "flock timeout"; exit 3; }
exec 9>"$HOME/spd_qb2_chip$MATE.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
for arm in A P; do
  cp $WT/perf/rfd4/p512/call_* $R/tree-$arm/perf/rfd4/p512/
  (cd $R/tree-$arm && bash perf/rfd4/p512/call_prof.sh $R/census/$arm $N --n 512 --batch 4 --lane 1 --nsteps 2) \
    > $R/census-$arm.log 2>&1
  stamp "census $arm rc=$?"
done
exec 8>&- 9>&-
$BOOK rm $OWN > /dev/null
stamp CENSUSDONE
