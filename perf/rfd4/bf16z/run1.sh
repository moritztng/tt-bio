#!/bin/bash
# bf16z run1: RFD4's pair stream in bf16 through the fused programs (`pair_bf16z`) on top of RFD4's default set.
# z / s vs float64 (zprec.py, 256 and 512 tokens), paired traced trunk A/B at 512 and 256 (perf/rfd4/bfp8/stack_ab.py,
# 30 reps, AICLK sampled during), a census at 512 per arm, then 31 seeds at 512 per arm graded per seed (grade.py)
# and as a population (quality.py).
#   perf/rfd4/bf16z/run1.sh <card> [legs]
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/bf16z/${RUN:-run1}; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-zprec256 ab512 ab256 census-base census-stack zprec512 design grade}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-bf16z PYTHONPATH=$W:$W/perf
mkdir -p $O/design $O/grade; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 14400 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 2>/dev/null | cut -c1-70) tree=$(cat TREE 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
S=$B,pair_bf16z
echo "base $B" >> $O/run.log
f() { ls $O/design/$1/pdl1_binder_512_s$2_r*_generated.cif | tail -1; }   # s0 also has the cold run r0
for leg in $LEGS; do
  case $leg in
    zprec*) n=${leg#zprec}; timeout 3600 $PY perf/rfd4/bf16z/zprec.py --n $n --out $O/$leg.json > $O/$leg.log 2>&1 ;;
    ab*) n=${leg#ab}; TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
           --out $O/$leg.json --n $n --reps 30 --base $B --levers $S > $O/$leg.log 2>&1 ;;
    census-base) TT_BIO_LEVERS=$B bash perf/rfd4/e2e/elt_prof.sh $O/$leg $C --n 512 > $O/$leg.log 2>&1 ;;
    census-stack) TT_BIO_LEVERS=$S bash perf/rfd4/e2e/elt_prof.sh $O/$leg $C --n 512 > $O/$leg.log 2>&1 ;;
    design) for a in base stack; do L=$([ $a = base ] && echo $B || echo $S)
              TT_BIO_LEVERS=$L timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
                --weights $HOME/rfd4-bundle --out $O/design/$a --rng upstream --seeds $(seq 0 30) > $O/design/$a.log 2>&1
              stamp "design $a rc=$?"; done ;;
    grade) exec 9>&-
           for k in $(seq 0 30); do $PY perf/rfd4/e2e/grade.py --ours $(f stack $k) --upstream $(f base $k) \
               --floor $(f base $(( (k+1)%31 ))) --out $O/grade/s$k.json >/dev/null 2>&1; done
           cp $HOME/scratch/rfd4-land5/QUALITY.base.json $O/quality512.json
           $PY perf/rfd4/quality.py grade --feature binder --tt $O/design/stack --ref $O/design/base --allow-no-oracle \
             --label "RFD4 default + pair_bf16z vs default, 512 PD-L1 binder, 31 seeds" \
             --out $O/quality512.json > $O/quality512.log 2>&1 ;;
  esac
  rc=$?; stamp "$leg rc=$rc"
  [ "$leg" = zprec256 ] && [ $rc != 0 ] && { stamp "ABORT: the lever does not run"; exit 4; }
done
stamp ALLDONE
