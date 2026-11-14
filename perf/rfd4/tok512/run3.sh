#!/bin/bash
# rfd4-perf-tok512 run3, quality of the token-side set (R, G, W; S is bit-identical) on a chip without a determinism marker: z / s vs float64 at 512 b1 and
# 224 b10 (bf16z/zprec.py, two cycles so the recycle projection runs), then seed-matched designs main's default vs + the set at main's widths (512 w1, 224 w10), graded per
# seed against the seed floor and as a pooled binder population (quality.py). Never per design: RFD4 re-rolls a
# trajectory on any numeric change.
#   bash perf/rfd4/tok512/run3.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; LEGS=${3:-zprec512 zprec224 designs grade}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=${OWN:-rfd4-perf-tok512}; LEV=${LEV:-tok_recycle_mm,dit_sigf32,tok_w8}
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
[ -e $HOME/spd/locks/chip$N.BAD ] && { stamp "chip $N carries chip$N.BAD, refusing a quality leg"; exit 2; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-2700} --note "tok512 run3 quality" > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(basename $WT)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))")
f() { ls $R/$1/pdl1_binder_$2_d$3.cif; }
for leg in $LEGS; do
  case $leg in
    zprec224) timeout 3000 $PY perf/rfd4/bf16z/zprec.py --n 224 --batch 10 --cycles 2 --lever $LEV --out $R/$leg.json > $R/$leg.log 2>&1 ;;
    zprec512) timeout 3000 $PY perf/rfd4/bf16z/zprec.py --n 512 --batch 1 --cycles 2 --lever $LEV --out $R/$leg.json > $R/$leg.log 2>&1 ;;
    designs) for s in 512:1:31 224:10:40; do IFS=: read n w k <<< "$s"
               for a in base tok; do L=$B; [ $a = tok ] && L=$B,$LEV
                 TT_BIO_LEVERS=$L timeout 3600 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-$n.yaml --weights $W \
                   --out $R/n$n-$a --width $w --designs $k > $R/n$n-$a.log 2>&1
                 stamp "designs n$n-$a rc=$?"; done; done ;;
    grade) exec 9>&-; $BOOK rm $OWN > /dev/null
           for n in 512 224; do
             mkdir -p $R/grade-$n $R/pool-$n/tt $R/pool-$n/ref
             for k in $(seq 0 30); do
               ln -sf $(f n$n-tok $n $k) $R/pool-$n/tt/; ln -sf $(f n$n-base $n $k) $R/pool-$n/ref/
               nice -n 19 $PY perf/rfd4/e2e/grade.py --ours $(f n$n-tok $n $k) --upstream $(f n$n-base $n $k) \
                 --floor $(f n$n-base $n $(( (k+1)%31 ))) --out $R/grade-$n/s$k.json > /dev/null 2>&1 || echo "grade $n s$k failed"
             done
             cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality-$n.json
             nice -n 19 $PY perf/rfd4/quality.py grade --feature binder --tt $R/pool-$n/tt --ref $R/pool-$n/ref \
               --allow-no-oracle --label "tok512 $n: default + $LEV vs default, PD-L1 binder, 31 seeds" \
               --out $R/quality-$n.json > $R/quality-$n.log 2>&1
             stamp "quality $n rc=$? $(tail -1 $R/quality-$n.log | cut -c1-200)"
           done ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN > /dev/null 2>&1
stamp ALLDONE
