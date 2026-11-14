#!/bin/bash
# t1dm run3: lever F (trimul_t1_fgate) as a stack with main (pair_bfp8z in): z / s vs float64 at 256 and 512
# (bf16z/zprec.py), 31 seeds at 512 per arm graded per seed (grade.py) and as a population (quality.py), then T1
# digests at the other models' keys with F forced on (T1_FG=1) here and off on main's tree ($OLD): equal = RFD4 only.
#   RUN=run3 perf/rfd4/t1dm/run3.sh <main tree> [legs]     (run from the tree; first free qb2 card, up to 10 h)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/perf/rfd4/t1dm/${RUN:-run3}; PY=$HOME/tt-bio-dev/env/bin/python
OLD=${1:?main tree}; LEGS=${2:-zprec256 zprec512 design grade xtree}; ME=rfd4-perf-t1dm; LEV=trimul_t1_fgate
mkdir -p $O/design $O/grade; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait any card, legs $LEGS"
end=$(( $(date +%s) + 36000 )); C=
while [ -z "$C" ] && [ $(date +%s) -lt $end ]; do
  for c in 2 1 3 0; do
    [ -n "$(python3 $HOME/spd/feed/book.py active $c --except $ME 2>/dev/null)" ] && continue
    exec 9>"$HOME/spd_qb2_chip$c.lock"
    flock -n 9 || { exec 9>&-; continue; }
    fuser "/dev/tenstorrent/$c" >/dev/null 2>&1 && { exec 9>&-; continue; }
    C=$c; break
  done
  [ -z "$C" ] && sleep 30
done
[ -z "$C" ] && { stamp TIMEOUT; exit 3; }
python3 $HOME/spd/feed/book.py add $ME $C 7200 --now --note "t1dm F zprec, 31-seed grade, xtree" >/dev/null
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:$ME PYTHONPATH=$W:$W/perf
stamp "card $C tree $(cat .sha) main $(cat $OLD/.sha)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N - {'$LEV'})))")
S=$B,$LEV
echo "base $B" >> $O/run.log
f() { ls $O/design/$1/pdl1_binder_512_s$2_r*_generated.cif | tail -1; }   # s0 also has the cold run r0
R="--c 384 --hidden 256 --w8 --fid HiFi2"
for leg in $LEGS; do
  case $leg in
    zprec*) n=${leg#zprec}; timeout 3600 $PY perf/rfd4/bf16z/zprec.py --n $n --lever $LEV --out $O/$leg.json > $O/$leg.log 2>&1 ;;
    design) for a in base stack; do L=$([ $a = base ] && echo $B || echo $S)
              TT_BIO_LEVERS=$L timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
                --weights $HOME/rfd4-bundle --out $O/design/$a --rng upstream --seeds $(seq 0 30) > $O/design/$a.log 2>&1
              stamp "design $a rc=$?"; done ;;
    grade) for k in $(seq 0 30); do $PY perf/rfd4/e2e/grade.py --ours $(f stack $k) --upstream $(f base $k) \
               --floor $(f base $(( (k+1)%31 ))) --out $O/grade/s$k.json >/dev/null 2>&1; done
           cp $HOME/scratch/rfd4-land5/QUALITY.base.json $O/quality512.json
           $PY perf/rfd4/quality.py grade --feature binder --tt $O/design/stack --ref $O/design/base --allow-no-oracle \
             --label "RFD4 default + trimul_t1_fgate vs default, 512 PD-L1 binder, 31 seeds" \
             --out $O/quality512.json > $O/quality512.log 2>&1 ;;
    xtree) for tree in $W $OLD; do
             FG=; [ $tree = $W ] && FG=,T1_FG=1
             (cd $tree; export PYTHONPATH=$tree
              echo "tree $tree$FG" >> $O/xtree_t1.jsonl
              t1() { timeout 900 $PY perf/spd_trimul/in_t1.py "$@" --reps 3 --out $O/xtree_t1.jsonl; }
              t1 $R --ns 512 --set T1_PSIG=1,T1_B8=1$FG
              t1 --c 256 --hidden 256 --w8 --fid HiFi3 --no-mask --ns 512 --set T1_PSIG=1$FG
              t1 --c 384 --hidden 384 --w8 --no-mask --ns 512 --set T1_PSIG=1$FG
              t1 --c 128 --hidden 128 --no-mask --ns 512 ${FG:+--set T1_FG=1}) >> $O/xtree.log 2>&1
           done ;;
  esac
  rc=$?; stamp "$leg rc=$rc"
  case $leg in zprec*) [ $rc != 0 ] && { stamp "ABORT: the lever does not run"; break; } ;; esac
done
exec 9>&-; python3 $HOME/spd/feed/book.py rm $ME
stamp ALLDONE
