#!/bin/bash
# rfd4-perf-small device pass 5: re-measure docs/rfd4-design.md's cost table on this tree, CAND vs the default
# (CAND + trimul_t1_dram), family by family with the two arms back to back. Same commands as perf/rfd4/cond/runs.jsonl:
# 2 designs, seed 0, s/design = wall / 2 (compile included), AICLK and load1 sampled during each leg (clocked.py).
#   dev5.sh <chip> <tree> <out> <inputs dir> [families]
set -u
C=$1; T=$2; R=$3; IN=$4
FAMS=${5:-motif_scaffold structure_prediction_folding enzyme unconditional binder nanobody_cdr antibody_cdr}
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
S=$T/perf/rfd4/cond/specs
mkdir -p $R $T/perf/rfd4/cond/inputs && cp $IN/* $T/perf/rfd4/cond/inputs/
cp $HOME/rfd4-inputs/pdl1-128.yaml $HOME/rfd4-inputs/5o45_cropped.cif.zst $R/
args() {
  case $1 in
    binder128) echo "$R/pdl1-128.yaml --recipe thinking_medium";;
    unconditional) echo "$IN/monomer-short.cif.zst --recipe thinking_medium --annotated_cif";;
    binder) echo "$S/binder.yaml --recipe medium/front_loaded";;
    antibody_cdr) echo "$S/antibody.yaml --recipe high/front_loaded";;
    nanobody_cdr) echo "$IN/mzb1_7AAH__VHH_caplacizumab_cdr123.cif.zst --recipe high/front_loaded --annotated_cif";;
    enzyme) echo "$IN/M0040_13pk.cif.zst --recipe medium/midpeak --annotated_cif";;
    motif_scaffold) echo "$IN/1QJG_AA.cif.zst --recipe medium/midpeak --annotated_cif";;
    structure_prediction_folding) echo "$S/folding.yaml --recipe thinking_medium";;
  esac
}
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 10800 9 || { echo "no lock $(date -u +%FT%TZ)" >> $R/done; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=${LEASE_CARDS:-$C} TT_BIO_LEASE_HOLDER=worker:rfd4-perf-small PYTHONPATH=$T:$T/perf
cd $T
for f in $FAMS; do for arm in cand dflt; do
  if [ $arm = cand ]; then export TT_BIO_LEVERS=$CAND; else unset TT_BIO_LEVERS; fi
  j=$(timeout 2400 $PY perf/rfd4/small/clocked.py $R/$f-$arm.log $PY -m tt_bio.main design --model rfd4 $(args $f) \
      --out_dir $R/$f-$arm --seed 0 --num_designs 2)
  echo "{\"family\": \"$f\", \"arm\": \"$arm\", \"tokens\": \"$(grep -m1 -o '[0-9]* tokens' $R/$f-$arm.log)\", \"r\": ${j:-null}}" >> $R/runs.jsonl
done; done
echo "ALLDONE $(date -u +%FT%TZ)" >> $R/done
