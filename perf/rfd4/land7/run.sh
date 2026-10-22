#!/bin/bash
# rfd4-land7 device chain on one chip: device tests, paired A/B normal vs fast at 512, board cells, census v3.
#   flock ~/spd_qb2_chip<N>.lock bash perf/rfd4/land7/run.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle
mkdir -p $R; cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=0,$N TT_BIO_LEASE_HOLDER=worker:rfd4-land7 PYTHONPATH=$WT
FAST=$($PY -c "from tt_bio.rfd4.trunk import FAST_LEVERS as F; print(\",\".join(sorted(F)))")
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "start chip $N sha $(git rev-parse HEAD) fast=$FAST"
for leg in ${LEGS:-devtests ab512 d128 d256 d512 f512 census512 census128}; do
  case $leg in
    devtests) timeout 3000 $PY -m pytest -q -p no:cacheprovider -m device --junitxml=$R/devtests.xml \
                tests/test_rfd4_*.py tests/test_trimul_*.py ;;
    ab*) n=${leg#ab}; TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $W \
                --out $R/$leg.json --n $n --reps 30 ;;
    d*) n=${leg#d}; timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-$n.yaml --weights $W \
                --out $R/$leg --designs 6 --rng upstream ;;
    f*) n=${leg#f}; TT_BIO_LEVERS=$FAST timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-$n.yaml \
                --weights $W --out $R/$leg --designs 6 --rng upstream ;;
    census*) n=${leg#census}; bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
stamp ALLDONE
