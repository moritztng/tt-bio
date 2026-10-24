#!/bin/bash
# tok512 run2: run1's set plus W (tok_w8, bfp8 token weights at HiFi2), paired against the same tree with all of it off.
#   abN      paired traced trunk at N tokens and RFD4's served width, the whole set, 30 reps
#   ab512W   W alone at 512
#   wall512  whole designs at 512 (PD-L1 binder), base and set alternated WALL_K (2) times, 4 seeds each: the trunk
#            pass legs do not see the recycle projection (R), which runs between passes
# Run it from a snapshot of the tree (git archive into ~/scratch), never the live worktree.
#   bash perf/rfd4/tok512/run2.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tok512
OFF="rfd4.trunk.SDPA_FILL=False"; SET="+tok_recycle_mm,dit_sigf32,tok_w8"
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-3600} --note "tok512 run2: paired trunk + wall 512")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-3600} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat TREE 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))" 2>/dev/null | tail -1)
ab() {   # leg n batch levers base-set
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
    --out $R/$1.json --n $2 --batch $3 --reps 30 --base $B --levers "$4" --base-set "$5"
}
OFFPY="import runpy, sys, tt_bio.rfd4.trunk as t
t.SDPA_FILL = False
sys.argv[0] = 'perf/rfd4/e2e/design.py'; runpy.run_path(sys.argv[0], run_name='__main__')"
D="--spec $HOME/rfd4-inputs/pdl1-512.yaml --weights $HOME/rfd4-bundle --rng upstream --seeds 0 1 2 3"
for leg in ${3:-ab512 ab224 ab256 ab128 ab512W wall512}; do
  case $leg in
    ab512) ab $leg 512 1 "$SET" "$OFF" ;;
    ab224) ab $leg 224 10 "$SET" "$OFF" ;;
    ab256) ab $leg 256 4 "$SET" "$OFF" ;;
    ab128) ab $leg 128 16 "$SET" "$OFF" ;;
    ab512W) ab $leg 512 1 "+tok_w8" "" ;;
    wall512) for k in $(seq ${WALL_K:-2}); do
        TT_BIO_LEVERS=$B timeout 1800 $PY -c "$OFFPY" $D --out $R/wall512/base$k; stamp "wall base$k rc=$?"
        TT_BIO_LEVERS=$B,${SET#+} timeout 1800 $PY perf/rfd4/e2e/design.py $D --out $R/wall512/lev$k; stamp "wall lev$k rc=$?"
      done ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
