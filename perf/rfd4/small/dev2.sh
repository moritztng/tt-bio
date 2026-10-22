#!/bin/bash
# rfd4-perf-small device pass 2: grade trimul_t1_dram as a stack on a 224-token design (it does not act at 128 or 256).
#   dev2.sh <chip> <tree> <out>
# 9 designs per arm (r0 = compile, r1..r8 graded), per_design rng, PD-L1 + 109-residue binder, CAND vs CAND +
# trimul_t1_dram vs CAND again (A/A), AICLK sampled inside design.py; then the seed-matched CA shift against the
# seed floor and the oracle-free binder grade.
set -u
C=$1; T=$2; R=$3
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
G=$R/g224; mkdir -p $G
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
sed -e "s/size 128 total tokens (115 target residues + 13 binder residues)/size 224 total tokens (115 target residues + 109 binder residues)/" \
    -e "s/pdl1_binder_128:/pdl1_binder_224:/" -e "s/\"13-13, /\"109-109, /" $HOME/rfd4-inputs/pdl1-128.yaml > $R/pdl1-224.yaml
cp $HOME/rfd4-inputs/5o45_cropped.cif.zst $R/
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 10800 9 || { stamp "no lock"; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=${LEASE_CARDS:-$C} TT_BIO_LEASE_HOLDER=worker:rfd4-perf-small PYTHONPATH=$T:$T/perf
cd $T
for arm in cand dram cand2; do
  L=$CAND; [ $arm = dram ] && L=$CAND,trimul_t1_dram
  TT_BIO_LEVERS=$L timeout 5400 $PY perf/rfd4/e2e/design.py --spec $R/pdl1-224.yaml --weights $W \
    --out $G/224-${arm}1 --rng per_design --seed 0 --designs 9 --reseed > $G/224-$arm.log 2>&1
  stamp "design-$arm rc=$?"
done
exec 9>&-
cd $T/perf/rfd4/e2e
$PY resid_spread.py $G --base cand --arm dram --first 1 --out $R/spread224-dram-vs-cand.json > $R/spread224.log 2>&1
stamp "spread rc=$?"
$PY resid_spread.py $G --base cand --arm cand2 --first 1 --out $R/spread224-cand2-vs-cand.json > $R/spread224-aa.log 2>&1
stamp "spread-aa rc=$?"
mkdir -p $R/q/dram $R/q/cand
for a in dram cand; do for f in $G/224-${a}1/*_r*_generated.cif; do case $f in *_r0_*) ;; *) ln -sf $f $R/q/$a/;; esac; done; done
cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality224.json
cd $T && $PY perf/rfd4/quality.py grade --feature binder --tt $R/q/dram --ref $R/q/cand --allow-no-oracle \
  --label "CAND + trimul_t1_dram vs CAND, 224-token PD-L1 binder, 8 seeds" --out $R/quality224.json > $R/quality224.log 2>&1
stamp "quality rc=$?"
stamp ALLDONE
