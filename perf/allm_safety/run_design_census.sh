#!/bin/bash
# One design model, one card, under the _MM_BLOCK key census. The hook dumps every 3 s, so a
# run cut short still leaves its counts on disk -- which is the point: this measures which keys
# the model presents, not how long a design takes.
set -u
WT=/home/ttuser/.coworker/wt/allm-safety
MODEL=$1; CARD=$2; SPEC=$3
cd $WT
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-safety
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/allm_safety/out
exec $PY -u perf/allm_safety/mm_key_census.py \
  --python $PY --pythonpath $WT --label ${MODEL}-design-c${CARD} \
  --out $OUT/mmkeys_${MODEL}_design.json \
  -- -m tt_bio.main design $WT/$SPEC --model $MODEL --seed 0 \
     --out_dir $OUT/design_${MODEL}_c${CARD}
