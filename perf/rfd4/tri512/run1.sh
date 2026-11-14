#!/bin/bash
# tri512 run1 (pricing, main only): census v12 at 512 b1 and 384 b1, the triangle's programs split by stage skips.
#   t1        standalone T1 at RFD4's key as the trunk runs it (PSIG, B8, G8, FG, bfp8 weights and z, HiFi2): real,
#             compute only (ABL 112), DM only (527) and DM by stage (543 no z reads, 559 no gather, 591 no writes,
#             623 neither), 512 / 384
#   tail      the T2CM tail with GROW (main's default) against GROW off, and GROW with dm / abl skips, 512 / 384
#             (TL=... overrides the tail legs; run1b: each compute stage skip alone under GROW)
#   censusN   census v12, device ms a program a call (elt_prof + census_v2 + top_ops)
#   bash perf/rfd4/tri512/run1.sh <chip> <outdir> [legs]      (run from a snapshot tree with .sha)
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tri512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-3600} --note "tri512 run1: census v12 512/384, T1 + tail split")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-3600} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat .sha 2>/dev/null)"
G="grow=1"
TL=${TL:-"$G;$G,dm=1;$G,dm=2;$G,dm=12;$G,abl=27;$G,abl=27,dm=12"}
for leg in ${3:-t1 tail census512 census384}; do
  case $leg in
    t1) timeout 1500 $PY perf/spd_trimul/in_t1.py --c 384 --hidden 256 --w8 --z8 --no-mask --reps 20 --fid HiFi2 \
          --set T1_PSIG=1,T1_B8=1,T1_G8=1,T1_FG=1 --ns 512,384 --abl 112,527,543,559,591,623 --out $R/t1.jsonl ;;
    tail) timeout 1500 $PY perf/rfd4/g512/tail.py --ns 512,384 --legs "$TL" --out $R/tail.jsonl ;;
    census*) n=${leg#census}
      bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n \
        && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-$n-v12.json \
        && python3 perf/rfd4/land10/top_ops.py $R/$leg $R/top-ops-$n.json > $R/top-ops-$n.txt ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
