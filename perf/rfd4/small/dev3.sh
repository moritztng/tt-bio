#!/bin/bash
# rfd4-perf-small device pass 3: trimul_t1_dram (T1 + pair_tail on the DRAM route from 160 tokens) across the
# 160-352 buckets that CAND keeps in L1, T1 row-tile plans at 128, and census v3 at 128 with the fixed instrument.
#   dev3.sh <chip> <tree> <out>     tree = an exported checkout of wk/rfd4-perf-small
set -u
C=$1; T=$2; R=$3
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
mkdir -p $R
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 10800 9 || { stamp "no lock"; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=${LEASE_CARDS:-$C} TT_BIO_LEASE_HOLDER=worker:rfd4-perf-small PYTHONPATH=$T:$T/perf
cd $T
stamp start
for n in 160 192 224 256 288 320 352; do
  timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $W --out $R/stack-$n.json --n $n --reps 30 \
    --base $CAND --levers $CAND,trimul_t1_dram > $R/stack-$n.log 2>&1
  stamp "stack-$n rc=$?"
done
S="lever:${CAND//,/+}+trimul_t1_small;mask:none"
timeout 1200 $PY perf/rfd4/tri2/bench.py --weights $W --out $R/tri-128.json --n 128 --reps 30 \
  --arms "lever:${CAND//,/+};mask:none|$S|$S;set:trimul_tail.T1_ROWS=4|$S;set:trimul_tail.T1_ROWS=2|$S;set:trimul_tail.T1_ROWS=1" \
  > $R/tri-128.log 2>&1
stamp "tri-128 rc=$?"
TT_BIO_LEVERS=$CAND bash perf/rfd4/e2e/elt_prof.sh $R/census-cand $C --n 128 > $R/census-cand.log 2>&1
stamp "census-cand rc=$?"
stamp ALLDONE
