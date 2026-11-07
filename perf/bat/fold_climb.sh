#!/bin/bash
# bat-fold: does a JapanFold fold's diffusion sample axis batch on one chip, and what would two requests sharing
# a chip buy? One chip, PopVax's widths, the serving path (perf/mol1/fair/run.py: model loaded once, warm rounds,
# AICLK sampled in-run, foreign load recorded).
#   bash perf/bat/fold_climb.sh <chip> <outdir> <data dir: inputs/ + msa/cache>
# A cell is model:input:S:W = S diffusion samples denoised W at a time (--max_parallel_samples). Served default
# is S=5 W=5. Fold(S) = trunk + S x diffusion-per-sample(W), so S=1, S=5 W1, S=5 W5 and S=10 W10/W5 separate the
# trunk from the sample axis and read what widening the axis past one request's 5 would give.
set -u
C=$1; R=$2; D=$3; WT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-8edb023e00db/bin/python}
mkdir -p $R; R=$(cd $R && pwd)
stamp() { echo "$(date -u +%FT%TZ) $* load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
exec 9>"$HOME/spd_qb2_chip$C.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
stamp "start chip $C sha $(git -C $WT rev-parse --short HEAD)"
cd $D/inputs
for cell in ${CELLS:-opendde:9TH6:5:5 opendde:9TH6:1:1 opendde:9TH6:5:1 opendde:9TH6:10:10 opendde:9TH6:10:5 \
                     opendde:9DBP:5:5 opendde:9DBP:1:1 opendde:9DBP:5:1 opendde:9DBP:10:10 opendde:9DBP:10:5 \
                     mol1:9TH6:5:5 mol1:9TH6:1:1 mol1:9TH6:5:1 mol1:9TH6:10:10 \
                     mol1:9DBP:5:5 mol1:9DBP:1:1 mol1:9DBP:5:1 mol1:9DBP:10:10}; do
  IFS=: read m c s w <<< "$cell"
  o=$R/$m-$c-s$s-w$w; grep -qsx 0 $o.rc && continue
  TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bat-fold PYTHONPATH=$WT timeout ${T:-5400} \
    $PY $WT/perf/mol1/fair/run.py --chips $C --part dev --rounds ${ROUNDS:-2} --max-attempts 3 --model $m --arm exact \
    --input $D/inputs/$c.yaml --out $o --cli-extra --diffusion_samples $s --max_parallel_samples $w \
    --recycling_steps $([ $m = mol1 ] && echo 20 || echo 10) --msa_dir $D/msa/cache --msa_cache_only > $o.log 2>&1
  echo $? > $o.rc; stamp "$m $c s$s w$w rc=$(cat $o.rc)"
done
stamp ALLDONE
