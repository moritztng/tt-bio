#!/bin/bash
# g512 run2: GROW (trimul_tail.T2_GROW, on by default on this branch) graded on its own tree.
#   xtree     every T2CM caller form, GROW off vs on in one process, output sha at 256 and 1024 (xtree.py)
#   abN       paired traced trunk at N, RFD4's served set, base arm built with T2_GROW=False, 30 reps
#   wall512   whole designs at 512 (PD-L1 binder), base (T2_GROW off) and GROW alternated twice, 4 seeds each
#   bash perf/rfd4/g512/run2.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-512g
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-7200} --note "g512 run2: GROW xtree, paired trunk 512/384/256/224, wall 512")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-7200} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat TREE 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))")
OFF="import runpy, sys; from tt_bio import trimul_tail as T; T.T2_GROW = False; sys.argv[0] = 'perf/rfd4/e2e/design.py'; runpy.run_path(sys.argv[0], run_name='__main__')"
D="--spec $HOME/rfd4-inputs/pdl1-512.yaml --weights $HOME/rfd4-bundle --rng upstream --seeds 0 1 2 3"
for leg in ${3:-xtree ab512 ab384 ab256 ab224 wall512}; do
  case $leg in
    xtree) timeout 2400 $PY perf/rfd4/g512/xtree.py --ns 256,1024 --out $R/xtree.jsonl ;;
    ab*) n=${leg#ab}
      TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
        --out $R/$leg.json --n $n --reps 30 --base $B --levers $B --base-set "trimul_tail.T2_GROW=False" ;;
    wall512) for k in 1 2; do
        timeout 1800 $PY -c "$OFF" $D --out $R/wall512/base$k; stamp "wall base$k rc=$?"
        timeout 1800 $PY perf/rfd4/e2e/design.py $D --out $R/wall512/grow$k; stamp "wall grow$k rc=$?"
      done ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
