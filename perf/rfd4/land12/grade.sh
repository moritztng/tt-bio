#!/bin/bash
# rfd4-land12 grade from a run.sh outdir (CPU only, niced): designed-chain CA RMSD per seed by grade.py and the
# oracle-free binder grade by quality.py, 31 seeds, for 512 smain vs sbase (floor = sbase seed k vs k+1), 128 w16 vs
# w1 and 256 w4 vs w1; then --fast vs default ATOM identity on seeds 0-7.
#   bash perf/rfd4/land12/grade.sh <run outdir>
set -u
R=${1:?outdir}; WT=$(cd "$(dirname "$0")/../../.." && pwd); cd $WT
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python; export PYTHONPATH=$WT
s() { ls $R/$1/pdl1_binder_512_s$2_r*_generated.cif | tail -1; }   # s0 also carries the compile run r0
f() { ls $R/$1/pdl1_binder_${1:1:3}_d$2.cif; }
pool() {  # name, ours-file-fn, ours, ref, label
  local name=$1 fn=$2 ours=$3 ref=$4
  mkdir -p $R/grade-$name $R/pool-$name/tt $R/pool-$name/ref
  for k in $(seq 0 30); do
    ln -sf $($fn $ours $k) $R/pool-$name/tt/; ln -sf $($fn $ref $k) $R/pool-$name/ref/
    nice -n 19 $PY perf/rfd4/e2e/grade.py --ours $($fn $ours $k) --upstream $($fn $ref $k) \
      --floor $($fn $ref $(( (k+1)%31 ))) --out $R/grade-$name/s$k.json > /dev/null 2>&1 || echo "grade $name s$k failed"
  done
  cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality-$name.json
  nice -n 19 $PY perf/rfd4/quality.py grade --feature binder --tt $R/pool-$name/tt --ref $R/pool-$name/ref \
    --allow-no-oracle --label "land12 $name: $ours vs $ref, PD-L1 binder, 31 seeds" --out $R/quality-$name.json \
    > $R/quality-$name.log 2>&1
  echo "quality $name rc=$? $(tail -1 $R/quality-$name.log | cut -c1-200)"
}
pool stack s smain sbase
pool w16 f n128-w16 n128-w1
pool w4 f n256-w4 n256-w1
: > $R/fast-identity.txt
for k in $(seq 0 7); do
  echo "s$k $(cmp -s <(grep ^ATOM $(s sfast $k)) <(grep ^ATOM $(s smain $k)) && echo identical || echo DIFFERS)" >> $R/fast-identity.txt
done
