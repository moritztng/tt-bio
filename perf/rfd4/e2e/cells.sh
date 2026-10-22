#!/bin/bash
# rfd4-cells card leg: the 256 first-call probe, then whole designs at 128/256/512 on this tree
# (one cold + >=3 warm each, AICLK sampled inside every design, load beside it), then a batch-2 rung at 256.
#   bash perf/rfd4/e2e/cells.sh <chip> <outdir>     (run under flock ~/spd_qb2_chip<chip>.lock)
set -u
N=${1:?chip}; R=${2:?outdir}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle
mkdir -p $R; cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-cells PYTHONPATH=$WT
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "start chip $N sha $(git rev-parse HEAD) py $PY"
for leg in ${LEGS:-probe256 d128 d256 d512 b256}; do
  case $leg in
    probe*) n=${leg#probe}; timeout 2400 $PY perf/rfd4/e2e/first_call.py --spec $IN/pdl1-$n.yaml --weights $W \
              --out $R/$leg --designs 3 --rng ${RNG:-upstream} ;;
    pdram*) n=${leg#pdram}; timeout 2400 $PY perf/rfd4/e2e/first_call.py --spec $IN/pdl1-$n.yaml --weights $W \
              --out $R/$leg --designs 3 --rng ${RNG:-upstream} --dram $n ;;
    d*) n=${leg#d}; timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-$n.yaml --weights $W \
              --out $R/$leg --designs 4 --rng upstream ;;
    b*) n=${leg#b}; timeout 3600 $PY perf/rfd4/batch/sweep.py --spec $IN/pdl1-$n.yaml --weights $W \
              --out $R/$leg --ladder 1,2 --designs 2 --reps 1 ;;
    suites) timeout 3000 $PY -m pytest -q -p no:cacheprovider --junitxml=$R/suites.xml tests/test_rfd4_trunk_parity.py \
              tests/test_rfd4_trunk_golden.py tests/test_rfd4_sampler.py tests/test_rfd4_identity_mask.py tests/test_rfd4_first_call.py ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
stamp ALLDONE
