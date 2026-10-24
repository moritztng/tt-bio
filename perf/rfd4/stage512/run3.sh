#!/bin/bash
# stage512 run3: ROT (bmm_res writes x page-rotated, the moved T2 reader walks every bank a unit) against
# main's plain order, paired traced trunk at N, RFD4's served set, base arm bmm_res.ROT=False, 30 interleaved reps.
#   bash perf/rfd4/stage512/run3.sh <chip> <outdir> [legs]      (run from a snapshot tree with .sha)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-stage512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-7200} --note "stage512 run3: ROT paired trunk 512/384/256/224")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-7200} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat .sha 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))")
for leg in ${3:-ab512 ab384 ab256 ab224}; do
  n=${leg#ab}
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
    --out $R/$leg.json --n $n --reps 30 --base $B --levers $B --base-set "bmm_res.ROT=False" > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
