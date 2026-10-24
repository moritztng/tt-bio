#!/bin/bash
# b512 run2: the levers run1 kept, on by default in this tree, graded against the same tree with BASESET (default:
# outer_residual.GRAN=4 and trimul_tail.T1_WFL=0, the values on main) held in the base arm.
#   abN       paired traced trunk at N, RFD4's served set, 30 reps
#   wall512   whole designs at 512 (PD-L1 binder), base and lever alternated WALL_K (2) times, 4 seeds each
#   t1x       T1 digests at every model key with T1_WFL off and on (shared kernel, D-orch-140)
# Run it from a snapshot of the tree (git archive into ~/scratch), never the live worktree.
#   bash perf/rfd4/b512/run2.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-512b
BASESET=${BASESET:-outer_residual.GRAN=4;trimul_tail.T1_WFL=0}
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-7200} --note "b512 run2: paired trunk 512/384/256/224, wall 512")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-7200} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat TREE 2>/dev/null) baseset $BASESET"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))")
OFF="import runpy, sys, importlib
for one in '$BASESET'.split(';'):
    k, v = one.split('=', 1); m, a = k.rsplit('.', 1); setattr(importlib.import_module('tt_bio.' + m), a, eval(v))
sys.argv[0] = 'perf/rfd4/e2e/design.py'; runpy.run_path(sys.argv[0], run_name='__main__')"
D="--spec $HOME/rfd4-inputs/pdl1-512.yaml --weights $HOME/rfd4-bundle --rng upstream --seeds 0 1 2 3"
T1R="--w8 --no-mask --reps 3 --ns 512,256"
for leg in ${3:-ab512 ab384 ab256 ab224 wall512 t1x}; do
  case $leg in
    ab*) n=${leg#ab}
      TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
        --out $R/$leg.json --n $n --reps 30 --base $B --levers $B --base-set "$BASESET" ;;
    wall512) for k in $(seq ${WALL_K:-2}); do
        timeout 1800 $PY -c "$OFF" $D --out $R/wall512/base$k; stamp "wall base$k rc=$?"
        timeout 1800 $PY perf/rfd4/e2e/design.py $D --out $R/wall512/lev$k; stamp "wall lev$k rc=$?"
      done ;;
    t1x) for k in "--c 384 --hidden 256 --z8 --fid HiFi2 --set T1_PSIG=1,T1_B8=1,T1_G8=1,T1_FG=1" \
                  "--c 256 --hidden 256 --fid HiFi3" "--c 384 --hidden 384" "--c 128 --hidden 128"; do
        timeout 900 $PY perf/spd_trimul/in_t1.py $k $T1R --ab T1_WFL=1 --out $R/t1x.jsonl; done ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
