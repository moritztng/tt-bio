#!/bin/bash
# rfd4-perf-224 lever C quality leg (and the rfd4 UX gate leg first): seed-matched designs at auto_width and at twice it, on a chip without a
# determinism marker, then the b128 grade (designed-chain CA RMSD per seed against the seed floor, and the binder
# population grade). Width already moves bits on main, so the grade is pooled over 31 seeds, never per design.
#   bash perf/rfd4/perf-224/grade.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-224-grade
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-224 PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
[ -e $HOME/spd/locks/chip$N.BAD ] && { stamp "chip $N carries chip$N.BAD, refusing a quality leg"; exit 2; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N 3600 --note "perf-224 lever C grade" > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
# the release gate's rfd4 UX leg on this tree (the 'Designing ...' headline fix), clocked
UXPY=${UXPY:-$HOME/scratch/shipflow/venv312-2fae3a713044/bin/python}
timeout 2400 $UXPY perf/rfd4/perf-224/clocked.py $R/ux.json $UXPY scripts/ux_regression.py --model rfd4 > $R/ux.log 2>&1
stamp "ux rc=$?"
# designs a multiple of both widths, so every lane runs full
for leg in n224-w5:40 n224-w10:40 n192-w7:42 n192-w14:42; do
  IFS=: read name k <<< "$leg"; n=${name:1:3}; w=${name##*w}
  timeout 2400 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-$n.yaml --weights $W --out $R/$name --width $w \
    --designs $k > $R/$name.log 2>&1
  stamp "$name rc=$?"
done
exec 9>&-; $BOOK rm $OWN > /dev/null; stamp "designs done"
f() { ls $R/$1/pdl1_binder_${1:1:3}_d$2.cif; }
for g in c224:n224-w10:n224-w5 c192:n192-w14:n192-w7; do
  IFS=: read name ours ref <<< "$g"
  mkdir -p $R/grade-$name $R/pool-$name/tt $R/pool-$name/ref
  for k in $(seq 0 30); do
    ln -sf $(f $ours $k) $R/pool-$name/tt/; ln -sf $(f $ref $k) $R/pool-$name/ref/
    nice -n 19 $PY perf/rfd4/e2e/grade.py --ours $(f $ours $k) --upstream $(f $ref $k) \
      --floor $(f $ref $(( (k+1)%31 ))) --out $R/grade-$name/s$k.json > /dev/null 2>&1 || echo "grade $name s$k failed"
  done
  cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality-$name.json
  nice -n 19 $PY perf/rfd4/quality.py grade --feature binder --tt $R/pool-$name/tt --ref $R/pool-$name/ref \
    --allow-no-oracle --label "perf-224 $name: $ours vs $ref, PD-L1 binder, 31 seeds" --out $R/quality-$name.json \
    > $R/quality-$name.log 2>&1
  stamp "quality $name rc=$? $(tail -1 $R/quality-$name.log | cut -c1-200)"
done
stamp ALLDONE
