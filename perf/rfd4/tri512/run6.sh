#!/bin/bash
# tri512 run6: S1K (trimul_tail.T2_S1K, the moved T2 reader reads a P = 2 part's stage tile as one 1 KB read) against
# main's two 512 B reads, paired traced trunk at N, RFD4's served set, base arm T2_S1K=False, 30 interleaved reps.
#   bash perf/rfd4/tri512/run6.sh <chip> <outdir> [legs]      (run from a snapshot tree with .sha)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tri512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-7200} --note "tri512 run6: S1K paired trunk 512/384/256/224")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-7200} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat .sha 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))")
for leg in ${3:-ab512 ab384 ab256 ab224}; do
  n=${leg#ab}
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
    --out $R/$leg.json --n $n --reps 30 --base $B --levers $B --base-set "trimul_tail.T2_S1K=False" > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
