#!/bin/bash
# rfd4-perf-small device pass 2: grade trimul_t1_small as a stack on the 128-token design, after dev1 says it pays.
#   dev2.sh <chip> <tree> <out>
# 9 designs per arm (r0 = compile, r1..r8 graded), per_design rng, PD-L1 binder at 128, CAND vs CAND + trimul_t1_small,
# AICLK sampled inside design.py; then the seed-matched CA shift against the seed floor and the oracle-free binder grade.
set -u
C=$1; T=$2; R=$3
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
G=$R/g128; mkdir -p $G
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 10800 9 || { stamp "no lock"; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-small PYTHONPATH=$T:$T/perf
cd $T
for arm in cand small cand2; do
  L=$CAND; [ $arm = small ] && L=$CAND,trimul_t1_small
  TT_BIO_LEVERS=$L timeout 3600 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-128.yaml --weights $W \
    --out $G/128-${arm}1 --rng per_design --seed 0 --designs 9 --reseed > $G/128-$arm.log 2>&1
  stamp "design-$arm rc=$?"
done
exec 9>&-
cd $T/perf/rfd4/e2e
$PY resid_spread.py $G --base cand --arm small --first 1 --out $R/spread128-small-vs-cand.json > $R/spread128.log 2>&1
stamp "spread rc=$?"
$PY resid_spread.py $G --base cand --arm cand2 --first 1 --out $R/spread128-cand2-vs-cand.json > $R/spread128-aa.log 2>&1
stamp "spread-aa rc=$?"
mkdir -p $R/q/small $R/q/cand
for a in small cand; do for f in $G/128-${a}1/*_r*_generated.cif; do case $f in *_r0_*) ;; *) ln -sf $f $R/q/$a/;; esac; done; done
cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality128.json
cd $T && $PY perf/rfd4/quality.py grade --feature binder --tt $R/q/small --ref $R/q/cand --allow-no-oracle \
  --label "CAND + trimul_t1_small vs CAND, 128-token PD-L1 binder, 8 seeds" --out $R/quality128.json > $R/quality128.log 2>&1
stamp "quality rc=$?"
stamp ALLDONE
