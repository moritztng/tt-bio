#!/bin/bash
# Interleaved fold A/B for the derived fused _MM_BLOCK key, on ONE card: the one this row holds.
#
# The lease is the card itself and nothing else. The earlier form was `LEASE=2; [ $CARD != 2 ] &&
# LEASE=2,$CARD`, which widened every run onto card 2 whatever card was asked for. That is how a run
# on 2026-09-21 opened card 2 while `of3t-crop640` held its lease, took the benchlock away from
# `allm-audit` for 30 minutes, and voided its own measurement. A grant is one card; widen it only
# for a deliberate fanout, on that command, after reading the sibling's lease file.
WT=/home/ttuser/.coworker/wt/allm-gates
cd $WT || exit 1
CARD=$1; MODEL=$2; ARMS=$3
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=${LEASE_CARDS:-$CARD} TT_BIO_LEASE_HOLDER=worker:allm-gates
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh allm-gates -- \
  $PY -u perf/allm_gates/fused_key_ab.py --model $MODEL --size 512 --arms $ARMS \
      --out perf/allm_gates/ab_${MODEL}_512_qb2c${CARD}.json
rc=$?
echo "RC=$rc for $MODEL"
echo "ABDONE $MODEL $(date -u +%FT%TZ)"
exit $rc
