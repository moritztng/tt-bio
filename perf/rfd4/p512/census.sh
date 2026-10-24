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
held() { fuser /dev/tenstorrent/$N /dev/tenstorrent/$MATE 2>/dev/null | grep -o '[0-9]\+' | tr '\n' ' '; }
gate() { pgrep -af "shipflow|RELEASE_GATE_RUN_DIR|release_gate|gate_fanout" | grep -v -e pgrep -e book.py |
  grep -qE "(TT_VISIBLE_DEVICES=|spd_qb2_chip)($N|$MATE)[^0-9]"; }
for arm in A P; do
  q=0; while [ $q -lt 6 ]; do if gate || [ -n "$(held)" ]; then q=0; else q=$((q + 1)); fi; sleep 20; done
  exec 8>"$HOME/spd_qb2_chip$N.lock" 9>"$HOME/spd_qb2_chip$MATE.lock"; flock -w 1800 8 && flock -w 1800 9
  cp $WT/perf/rfd4/p512/call_* $R/tree-$arm/perf/rfd4/p512/
  (cd $R/tree-$arm && bash perf/rfd4/p512/call_prof.sh $R/census/$arm $N --n 512 --batch 4 --lane 1 --nsteps 2) \
    > $R/census-$arm.log 2>&1
  rc=$?; exec 8>&- 9>&-
  stamp "census $arm rc=$rc"
done
$BOOK rm $OWN > /dev/null
stamp CENSUSDONE
