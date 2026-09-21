#!/bin/bash
# one gate census per model, on the card named by $1, models in $2 (comma list)
WT=/home/ttuser/.coworker/wt/allm-gates
cd $WT || exit 1
CARD=$1; MODELS=$2
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
# lease is the card itself; see run_ab.sh for why widening it by default was a defect
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=${LEASE_CARDS:-$CARD} TT_BIO_LEASE_HOLDER=worker:allm-gates
PY=/home/ttuser/tt-bio-dev/env/bin/python3
for M in ${MODELS//,/ }; do
  echo "=== $M on card $CARD $(date -u +%FT%TZ) ==="
  $PY -u perf/allm_gates/gate_census.py --model $M --size 512 --folds 2 \
      --out perf/allm_gates/census_${M}_512_qb2c${CARD}.json
  echo "RC=$? for $M"
done
echo "ALLDONE card $CARD $(date -u +%FT%TZ)"
