#!/bin/bash
# rfd4-perf-tri128 device pass 1: pair_tail_small at 128 and 96 tokens, paired traced trunk passes against main's
# default set (30 reps, AICLK sampled during by stack_ab). Arm B: T2 reads its x and g operands from L1 instead of a
# DRAM copy (trimul_tail.T2_L1_IN).
#   run1.sh <chip> <tree> <out>     tree = an exported checkout of wk/rfd4-perf-tri128
set -u
C=$1; T=$2; R=$3
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
mkdir -p $R
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 10800 9 || { stamp "no lock"; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=${LEASE_CARDS:-$C} TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri128 PYTHONPATH=$T:$T/perf
cd $T
BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))")
stamp "start sha $(cat $T/SHA 2>/dev/null) base $BASE"
for n in 128 96; do
  timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $W --out $R/a-$n.json --n $n --reps 30 \
    --base $BASE --levers $BASE,pair_tail_small > $R/a-$n.log 2>&1
  stamp "a-$n rc=$?"
  timeout 2400 $PY -c "import runpy, sys; import tt_bio.trimul_tail as M; M.T2_L1_IN = True; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" \
    perf/rfd4/bfp8/stack_ab.py --weights $W --out $R/b-$n.json --n $n --reps 30 \
    --base $BASE --levers $BASE,pair_tail_small > $R/b-$n.log 2>&1
  stamp "b-$n rc=$?"
done
TT_BIO_LEVERS=$BASE,pair_tail_small bash perf/rfd4/e2e/elt_prof.sh $R/census-a $C --n 128 > $R/census-a.log 2>&1
stamp "census-a rc=$?"
stamp ALLDONE
