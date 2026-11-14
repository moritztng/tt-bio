#!/bin/bash
# rfd4-perf-b128 grade from a run3.sh / run5.sh outdir (CPU only, niced): per-seed designed-chain CA RMSD by grade.py and the
# oracle-free population grade by quality.py, 31 seeds. pb: 128 b16 pair_batch vs b16 default (same width); pb1: b16
# pair_batch vs b1 default; w16: b16 default vs b1 default, the width's own perturbation for scale; pb256: 256 b4
# pair_batch vs b4 default.
#   bash perf/rfd4/b128/grade.sh <run outdir>
set -u
R=${1:?outdir}; WT=$(cd "$(dirname "$0")/../../.." && pwd); cd $WT
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python; export PYTHONPATH=$WT
f() { ls $R/$1/pdl1_binder_${1:1:3}_d$2.cif; }
for g in pb:n128-w16-pb:n128-w16-base pb1:n128-w16-pb:n128-w1-base w16:n128-w16-base:n128-w1-base \
    pb256:n256-w4-pb:n256-w4-base; do
  IFS=: read name ours ref <<< "$g"
  mkdir -p $R/grade-$name $R/pool-$name/tt $R/pool-$name/ref
  for k in $(seq 0 30); do
    ln -sf $(f $ours $k) $R/pool-$name/tt/; ln -sf $(f $ref $k) $R/pool-$name/ref/
    nice -n 19 $PY perf/rfd4/e2e/grade.py --ours $(f $ours $k) --upstream $(f $ref $k) \
      --floor $(f $ref $(( (k+1)%31 ))) --out $R/grade-$name/s$k.json > /dev/null 2>&1 || echo "grade $name s$k failed"
  done
  cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality-$name.json
  nice -n 19 $PY perf/rfd4/quality.py grade --feature binder --tt $R/pool-$name/tt --ref $R/pool-$name/ref \
    --allow-no-oracle --label "b128 $name: $ours vs $ref, PD-L1 binder, 31 seeds" --out $R/quality-$name.json \
    > $R/quality-$name.log 2>&1
  echo "quality $name rc=$? $(tail -1 $R/quality-$name.log | cut -c1-200)"
done
