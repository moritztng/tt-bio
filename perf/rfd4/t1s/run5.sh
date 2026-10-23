#!/bin/bash
# t1s run5: lever C, a and b written as bfp8 at RFD4's 8-row items (trimul_t1b8; items paired per face pair,
# kernels/trimul_in_t1/gather.hpp), on top of RFD4's default (A, the split gather, included).
#   std    standalone T1, bf16 then bfp8 a / b, 512/384/256 no mask and 512 masked: rel rms vs float64 and ms
#   il     Boltz-2's key (IL 32 plan), bfp8 at IL 32 and at IL 8: digests must match (same values, other tiling)
#   ab512 ab256 ab384  paired traced trunk, base = default, stack = default + trimul_t1b8, 30 reps
#   design grade       31 seeds at 512 per arm, per-seed CA RMSD (grade.py) and population quality
#   perf/rfd4/t1s/run5.sh <card> [legs]     (run from the tree; books the card)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/perf/rfd4/t1s/run5; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-std il ab512 ab256 ab384 design grade}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1s PYTHONPATH=$W:$W/perf
mkdir -p $O/design $O/grade; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1s ]; do
  python3 $HOME/spd/feed/book.py add rfd4-perf-t1s $C 4800 --note "RFD4 bet 27 t1s run5 (bfp8 a/b)" >/dev/null 2>&1; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
R="--c 384 --hidden 256 --w8 --fid HiFi2"
t1() { local leg=$1; shift; timeout 1500 $PY perf/spd_trimul/in_t1.py "$@" --out $O/$leg.jsonl >> $O/$leg.log 2>&1; }
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
S=$B,trimul_t1b8
f() { echo $O/design/$1/pdl1_binder_512_s$2_r$(($2 + 1))_generated.cif; }   # r0 is the compile pass
for leg in $LEGS; do
  case $leg in
    std) for s in T1_PSIG=1 T1_PSIG=1,T1_B8=1 T1_PSIG=1 T1_PSIG=1,T1_B8=1; do
           t1 $leg $R --no-mask --ns 512,384,256 --reps 20 --set $s; done
         t1 $leg $R --ns 512 --set T1_PSIG=1; t1 $leg $R --ns 512 --set T1_PSIG=1,T1_B8=1 ;;
    il) for r in 32 8; do t1 $leg --c 128 --hidden 128 --no-mask --ns 512,256 --reps 3 --set T1_B8=1,T1_ROWS=$r; done ;;
    ab*) n=${leg#ab}; TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
           --out $O/$leg.json --n $n --reps 30 --base $B --levers $S > $O/$leg.log 2>&1 ;;
    design) for a in base stack; do L=$([ $a = base ] && echo $B || echo $S)
              TT_BIO_LEVERS=$L timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
                --weights $HOME/rfd4-bundle --out $O/design/$a --rng upstream --seeds $(seq 0 30) > $O/design/$a.log 2>&1
              stamp "design $a rc=$?"; done ;;
    grade) exec 9>&-; python3 $HOME/spd/feed/book.py rm rfd4-perf-t1s
           for k in $(seq 0 30); do $PY perf/rfd4/e2e/grade.py --ours $(f stack $k) --upstream $(f base $k) \
               --floor $(f base $(( (k+1)%31 ))) --out $O/grade/s$k.json >/dev/null 2>&1; done
           cp $HOME/scratch/rfd4-land5/QUALITY.base.json $O/quality512.json
           $PY perf/rfd4/quality.py grade --feature binder --tt $O/design/stack --ref $O/design/base --allow-no-oracle \
             --label "RFD4 default + trimul_t1b8 vs default, 512 PD-L1 binder, 31 seeds" \
             --out $O/quality512.json > $O/quality512.log 2>&1 ;;
  esac
  stamp "$leg rc=$?"
  # a T1 that cannot build or reads wrong stops the chain before the trunk and design legs spend the card
  if [ $leg = std ] && ! grep -q '"t1_a_rel_rms_f64"' $O/std.jsonl 2>/dev/null; then stamp "std produced no T1 record, stop"; break; fi
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1s
stamp ALLDONE
