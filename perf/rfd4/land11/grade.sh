#!/bin/bash
# rfd4-land11 grade from a run.sh outdir (CPU only, niced): per-seed designed-chain CA RMSD by grade.py and the
# oracle-free binder grade by quality.py, 31 seeds, for 128 w16 vs w1 and 256 w4 vs w1 on the same tree; the 8-seed
# 512 sanity against land10's designs (same seeds, same venv) by grade.py.
#   bash perf/rfd4/land11/grade.sh <run outdir> <land10 sanity dir>
set -u
R=${1:?outdir}; L10=${2:?land10 sanity}; WT=$(cd "$(dirname "$0")/../../.." && pwd); cd $WT
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python; export PYTHONPATH=$WT
f() { ls $R/$1/pdl1_binder_${1:1:3}_d$2.cif; }
for g in w16:n128-w16:n128-w1 w4:n256-w4:n256-w1; do
  IFS=: read name ours ref <<< "$g"
  mkdir -p $R/grade-$name $R/pool-$name/tt $R/pool-$name/ref
  for k in $(seq 0 30); do
    ln -sf $(f $ours $k) $R/pool-$name/tt/; ln -sf $(f $ref $k) $R/pool-$name/ref/
    nice -n 19 $PY perf/rfd4/e2e/grade.py --ours $(f $ours $k) --upstream $(f $ref $k) \
      --floor $(f $ref $(( (k+1)%31 ))) --out $R/grade-$name/s$k.json > /dev/null 2>&1 || echo "grade $name s$k failed"
  done
  cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality-$name.json
  nice -n 19 $PY perf/rfd4/quality.py grade --feature binder --tt $R/pool-$name/tt --ref $R/pool-$name/ref \
    --allow-no-oracle --label "land11 $name: $ours vs $ref, PD-L1 binder, 31 seeds" --out $R/quality-$name.json \
    > $R/quality-$name.log 2>&1
  echo "quality $name rc=$? $(tail -1 $R/quality-$name.log | cut -c1-200)"
done
s() { ls $1/pdl1_binder_512_s$2_r*_generated.cif | tail -1; }
mkdir -p $R/grade-sanity
for k in $(seq 0 7); do
  a=$(s $R/sanity $k); b=$(s $L10 $k)
  echo "s$k $(cmp -s <(grep ^ATOM $a) <(grep ^ATOM $b) && echo identical || echo DIFFERS)" >> $R/sanity-identity.txt
  nice -n 19 $PY perf/rfd4/e2e/grade.py --ours $a --upstream $b --floor $(s $L10 $(( (k+1)%8 ))) \
    --out $R/grade-sanity/s$k.json > /dev/null 2>&1 || echo "s$k grade failed" >> $R/sanity-identity.txt
done
