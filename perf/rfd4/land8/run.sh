#!/bin/bash
# rfd4-land8 device chain on one chip: device tests on main, then 31-seed 512 designs for 6e42015fc default,
# main default and main --fast (the stack grade and the fast grade), the paired traced fast A/B at 512, board
# cells at 128/256/512, the host/device split at 512 and census v5 at 512 and 128. Grading runs on the CPU after.
#   flock ~/spd_qb2_chip<N>.lock bash perf/rfd4/land8/run.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd); BASE=$WT/artifacts/land8/base6e4
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle
mkdir -p $R; cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-land8 PYTHONPATH=$WT
FAST=$($PY -c "from tt_bio.rfd4.trunk import FAST_LEVERS as F; print(','.join(sorted(F)))" 2>/dev/null)
S="--spec $IN/pdl1-512.yaml --weights $W --rng upstream --seeds $(seq 0 30)"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "start chip $N sha $(git rev-parse HEAD) py $PY fast=$FAST"
for leg in ${LEGS:-devtests sbase smain sfast ab512 d128 d256 d512 f512 split512 census512 census128}; do
  case $leg in
    devtests) timeout 3000 $PY -m pytest -q -p no:cacheprovider -m device --junitxml=$R/devtests.xml \
                tests/test_rfd4_*.py tests/test_trimul_*.py ;;
    sbase) (cd $BASE && PYTHONPATH=$BASE timeout 5400 $PY perf/rfd4/e2e/design.py $S --out $R/sbase) ;;
    smain) timeout 5400 $PY perf/rfd4/e2e/design.py $S --out $R/smain ;;
    sfast) TT_BIO_LEVERS=$FAST timeout 5400 $PY perf/rfd4/e2e/design.py $S --out $R/sfast ;;
    ab*) n=${leg#ab}; TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $W \
                --out $R/$leg.json --n $n --reps 30 ;;
    d*) n=${leg#d}; timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-$n.yaml --weights $W \
                --out $R/$leg --designs 6 --rng upstream ;;
    f*) n=${leg#f}; TT_BIO_LEVERS=$FAST timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-$n.yaml \
                --weights $W --out $R/$leg --designs 6 --rng upstream ;;
    split*) n=${leg#split}; timeout 2400 $PY perf/rfd4/e2e/host_split.py --spec $IN/pdl1-$n.yaml --weights $W \
                --out $R/$leg --designs 4 ;;
    census*) n=${leg#census}; bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
stamp ALLDONE
