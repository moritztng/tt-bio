#!/bin/bash
# rfd4-perf-tri128 device pass 2: pair_tail_small (T2 reading L1 operands) at 128 and 96 tokens.
#   run2.sh <chip> <tree> <out> [designs per arm, default 31]
# 1. module: both triangle calls on the float32 stream, ms a call incl. the residual write, update vs float64
# 2. trunk: paired traced passes vs the default set without pair_tail_small, 30 reps (stack_ab)
# 3. designs at 128 (PD-L1 + 13-residue binder, per_design rng): default, lever, default again (A/A); r0 is the
#    compile and is excluded; then the seed-matched CA shift vs the seed floor and the oracle-free binder grade
set -u
C=$1; T=$2; R=$3; D=${4:-31}
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
G=$R/g128; mkdir -p $G
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
cp $HOME/rfd4-inputs/pdl1-128.yaml $HOME/rfd4-inputs/5o45_cropped.cif.zst $R/
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 10800 9 || { stamp "no lock"; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=${LEASE_CARDS:-$C} TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri128 PYTHONPATH=$T:$T/perf
cd $T
BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L - {'pair_tail_small'})))")
stamp "start sha $(cat $T/SHA 2>/dev/null) base $BASE"
for n in 128 96; do
  timeout 1800 $PY perf/rfd4/tri128/module.py --weights $W --out $R/module-$n.json --n $n \
    --arms "$BASE|$BASE,pair_tail_small|$BASE" > $R/module-$n.log 2>&1
  stamp "module-$n rc=$?"
  timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $W --out $R/stack-$n.json --n $n --reps 30 \
    --base $BASE --levers $BASE,pair_tail_small > $R/stack-$n.log 2>&1
  stamp "stack-$n rc=$?"
done
for arm in base small base2; do
  L=$BASE; [ $arm = small ] && L=$BASE,pair_tail_small
  TT_BIO_LEVERS=$L timeout 5400 $PY perf/rfd4/e2e/design.py --spec $R/pdl1-128.yaml --weights $W \
    --out $G/128-${arm}1 --rng per_design --seed 0 --designs $D --reseed > $G/128-$arm.log 2>&1
  stamp "design-$arm rc=$?"
done
exec 9>&-
cd $T/perf/rfd4/e2e
$PY resid_spread.py $G --base base --arm small --first 1 --out $R/spread128-small-vs-base.json > $R/spread128.log 2>&1
stamp "spread rc=$?"
$PY resid_spread.py $G --base base --arm base2 --first 1 --out $R/spread128-aa.json > $R/spread128-aa.log 2>&1
stamp "spread-aa rc=$?"
mkdir -p $R/q/small $R/q/base
for a in small base; do for f in $G/128-${a}1/*_r*_generated.cif; do case $f in *_r0_*) ;; *) ln -sf $f $R/q/$a/;; esac; done; done
cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality128.json
cd $T && $PY perf/rfd4/quality.py grade --feature binder --tt $R/q/small --ref $R/q/base --allow-no-oracle \
  --label "default + pair_tail_small vs default, 128-token PD-L1 binder, $((D - 1)) seeds" --out $R/quality128.json > $R/quality128.log 2>&1
stamp "quality rc=$?"
stamp ALLDONE
