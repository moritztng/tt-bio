#!/bin/bash
WT=/home/ttuser/.coworker/wt/allm-gates
cd $WT || exit 1
CARD=$1; MODEL=$2; ARMS=$3
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
LEASE=2; [ "$CARD" != 2 ] && LEASE=2,$CARD
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$LEASE TT_BIO_LEASE_HOLDER=worker:allm-gates
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh allm-gates -- \
  $PY -u perf/allm_gates/fused_key_ab.py --model $MODEL --size 512 --arms $ARMS \
      --out perf/allm_gates/ab_${MODEL}_512_qb2c${CARD}.json
echo "RC=$? for $MODEL"
echo "ABDONE $MODEL $(date -u +%FT%TZ)"
