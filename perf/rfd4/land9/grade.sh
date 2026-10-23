#!/bin/bash
# rfd4-land9 grades from run.sh's 512 arms (CPU only): per-seed designed-chain CA RMSD by grade.py and the
# oracle-free population grade by quality.py for main default vs 288f1ed37 (stack), and for --fast the ATOM records
# of sfast against smain seed by seed (the two lever sets are the same object on main, so they must match exactly).
#   bash perf/rfd4/land9/grade.sh <run outdir>
set -u
R=${1:?outdir}; WT=$(cd "$(dirname "$0")/../../.." && pwd); cd $WT
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python; export PYTHONPATH=$WT
f() { ls $PWD/$R/$1/pdl1_binder_512_s$2_r$(($2+1))_generated.cif; }
name=stack; ours=smain; ref=sbase
mkdir -p $R/grade-$name $R/pool-$name/tt $R/pool-$name/ref
for k in $(seq 0 30); do
  ln -sf $(f $ours $k) $R/pool-$name/tt/; ln -sf $(f $ref $k) $R/pool-$name/ref/
  $PY perf/rfd4/e2e/grade.py --ours $(f $ours $k) --upstream $(f $ref $k) --floor $(f $ref $(( (k+1)%31 ))) \
    --out $R/grade-$name/s$k.json > /dev/null 2>&1 || echo "grade $name s$k failed"
done
cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality-$name.json
$PY perf/rfd4/quality.py grade --feature binder --tt $R/pool-$name/tt --ref $R/pool-$name/ref --allow-no-oracle \
  --label "land9 $name: $ours vs $ref, 512 PD-L1 binder, 31 seeds" --out $R/quality-$name.json > $R/quality-$name.log 2>&1
echo "quality $name rc=$? $(tail -1 $R/quality-$name.log | cut -c1-200)"
for k in $(seq 0 7); do
  cmp -s <(grep -E '^(ATOM|HETATM)' $(f sfast $k)) <(grep -E '^(ATOM|HETATM)' $(f smain $k)) \
    && echo "fast s$k identical" || echo "fast s$k DIFFERS"
done | tee $R/fast-identity.txt
