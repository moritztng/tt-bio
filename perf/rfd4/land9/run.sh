#!/bin/bash
# rfd4-land9 device chain on one chip: device tests on main, board cells at 512/256/128 (6 designs, first is cold),
# the host/device split and census v6 at 512, then 31-seed 512 designs for 288f1ed37 (land8 main) and main default
# (the stack grade), and 8 seeds of main with --fast's lever set named explicitly (FAST_LEVERS = NORMAL_LEVERS on
# main, so the fast check is output identity against smain). Grading runs on the CPU after.
#   flock ~/spd_qb2_chip<N>.lock bash perf/rfd4/land9/run.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd); BASE=$WT/artifacts/land9/base288
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle
mkdir -p $R; cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-land9 PYTHONPATH=$WT
FAST=$($PY -c "from tt_bio.rfd4.trunk import FAST_LEVERS as F; print(','.join(sorted(F)))" 2>/dev/null)
S="--spec $IN/pdl1-512.yaml --weights $W --rng upstream"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "start chip $N sha $(git rev-parse HEAD) py $PY fast=$FAST"
for leg in ${LEGS:-devtests d512 d256 d128 split512 census512 smain sbase sfast}; do
  case $leg in
    devtests) timeout 3000 $PY -m pytest -q -p no:cacheprovider -m device --junitxml=$R/devtests.xml \
                tests/test_rfd4_*.py tests/test_trimul_*.py ;;
    sbase) (cd $BASE && PYTHONPATH=$BASE timeout 5400 $PY perf/rfd4/e2e/design.py $S --seeds $(seq 0 30) --out $R/sbase) ;;
    smain) timeout 5400 $PY perf/rfd4/e2e/design.py $S --seeds $(seq 0 30) --out $R/smain ;;
    sfast) TT_BIO_LEVERS=$FAST timeout 2400 $PY perf/rfd4/e2e/design.py $S --seeds $(seq 0 7) --out $R/sfast ;;
    d*) n=${leg#d}; timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-$n.yaml --weights $W \
                --out $R/$leg --designs 6 --rng upstream ;;
    split*) n=${leg#split}; timeout 2400 $PY perf/rfd4/e2e/host_split.py --spec $IN/pdl1-$n.yaml --weights $W \
                --out $R/$leg --designs 4 ;;
    census*) n=${leg#census}; bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
stamp ALLDONE
