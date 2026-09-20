#!/bin/bash
# One model, one card: fold at the model's own shipped defaults under the _MM_BLOCK key census.
# Counts only -- this run makes no timing claim, and the host is shared.
set -u
WT=/home/ttuser/.coworker/wt/allm-safety
MODEL=$1; CARD=$2; SIZE=${3:-512}
cd $WT
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-safety
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/allm_safety/out
exec $PY -u perf/allm_safety/mm_key_census.py \
  --python $PY --pythonpath $WT --label ${MODEL}-${SIZE}-c${CARD} \
  --out $OUT/mmkeys_${MODEL}_${SIZE}.json \
  -- -m tt_bio.main predict $WT/perf/size512/fixtures/cdk2x2_${SIZE}.yaml \
     --model $MODEL --single_sequence --seed 0 \
     --out_dir $OUT/fold_${MODEL}_${SIZE}_c${CARD}
