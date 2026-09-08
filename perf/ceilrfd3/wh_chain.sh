#!/bin/bash
# Walk one card through a list of rungs, one process per rung. Records every rung, including
# the ones above a failure: a ceiling is the largest size below the FIRST failure, and deciding
# that needs the rungs above it measured rather than skipped.
set -u
WT=/home/cust-team/mthuening/rfd3whv
PY=/home/cust-team/mthuening/tt-bio/env/bin/python
CARD=$1; TAG=$2; shift 2
LOG=$WT/perf/ceilrfd3/whverify/$TAG.log
JL=$WT/perf/ceilrfd3/whverify/$TAG.jsonl
mkdir -p "$WT/perf/ceilrfd3/whverify/$TAG"
cd "$WT" || exit 1
for total in "$@"; do
  echo "[chain] $(date -Is) tag=$TAG card=$CARD total=$total start" >> "$LOG"
  env PYTHONPATH=$WT WH_TOTAL=$total WH_TAG=$TAG WH_HOST_THREADS=30 \
      WH_OUT_DIR=$WT/perf/ceilrfd3/whverify/$TAG WH_JSONL=$JL \
      TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=1,$CARD \
      TT_BIO_LEASE_HOLDER=worker:rfd3-swiglu-dram-resident-768-wh-verify \
      TT_BIO_LEASE_TIMEOUT=10 \
      "$PY" perf/ceilrfd3/wh_ladder.py >> "$LOG" 2>&1
  echo "[chain] $(date -Is) tag=$TAG card=$CARD total=$total rc=$?" >> "$LOG"
done
echo "[chain] $(date -Is) tag=$TAG DONE" >> "$LOG"
