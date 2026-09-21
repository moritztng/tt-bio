#!/bin/bash
# One model, one card: fold at the model's own shipped defaults under the _MM_BLOCK key census.
# Counts only -- this run makes no timing claim, and the host is shared.
set -u
WT=/home/ttuser/.coworker/wt/allm-safety
MODEL=$1; CARD=$2; SIZE=${3:-512}
cd $WT
# CARD GRANT. `TT_BIO_LEASE_CARDS` is the DISPATCHER'S record of what was handed out and
# `device_lease.granted_cards()` refuses any open outside it. Setting it from a positional
# argument is self-granting: it forges the permission slip that check reads, which is how this
# row took `land-standing`'s card 0 on 2026-09-21 and cost it a gate arm. These launchers spawn
# a child process, so the in-process guard (card_guard.py) cannot cover them -- the card is
# pinned to this row's grant here instead, and any other card is refused outright.
ROW_GRANT=1
if [ "$CARD" != "$ROW_GRANT" ]; then
  echo "run: card $CARD is outside this row's grant ($ROW_GRANT). TT_BIO_LEASE_CARDS is the" >&2
  echo "     dispatcher's record, not a knob a worker may set. Refusing." >&2
  exit 75
fi
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$ROW_GRANT TT_BIO_LEASE_HOLDER=worker:allm-safety
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/allm_safety/out
exec $PY -u perf/allm_safety/mm_key_census.py \
  --python $PY --pythonpath $WT --label ${MODEL}-${SIZE}-c${CARD} \
  --out $OUT/mmkeys_${MODEL}_${SIZE}.json \
  -- -m tt_bio.main predict $WT/perf/size512/fixtures/cdk2x2_${SIZE}.yaml \
     --model $MODEL --single_sequence --seed 0 \
     --out_dir $OUT/fold_${MODEL}_${SIZE}_c${CARD}
