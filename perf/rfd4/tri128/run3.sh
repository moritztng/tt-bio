#!/bin/bash
# rfd4-perf-tri128 device pass 3: trimul_xn_l1 alone and stacked on pair_tail_small, 128 and 96 tokens.
#   run3.sh <chip> <tree> <out>
# module.py (ms a call incl. the residual write, update vs float64) and stack_ab (paired traced trunk passes, 30
# reps) against main's default set.
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
  timeout 1800 $PY perf/rfd4/tri128/module.py --weights $W --out $R/module-$n.json --n $n \
    --arms "$BASE|$BASE,trimul_xn_l1|$BASE,pair_tail_small|$BASE,pair_tail_small,trimul_xn_l1" > $R/module-$n.log 2>&1
  stamp "module-$n rc=$?"
  for arm in xn stack; do
    L=$BASE,trimul_xn_l1; [ $arm = stack ] && L=$BASE,pair_tail_small,trimul_xn_l1
    timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $W --out $R/$arm-$n.json --n $n --reps 30 \
      --base $BASE --levers $L > $R/$arm-$n.log 2>&1
    stamp "$arm-$n rc=$?"
  done
done
stamp ALLDONE
