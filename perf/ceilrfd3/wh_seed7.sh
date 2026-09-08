#!/bin/bash
set -u
WT=/home/cust-team/mthuening/rfd3whv
PY=/home/cust-team/mthuening/tt-bio/env/bin/python
cd "$WT" || exit 1
for t in 768 832 896; do
  env PYTHONPATH=$WT WH_TOTAL=$t WH_TAG=seed7 WH_SEED=7 WH_HOST_THREADS=30 \
      WH_OUT_DIR=$WT/perf/ceilrfd3/whverify/seed7 \
      WH_JSONL=$WT/perf/ceilrfd3/whverify/seed7.jsonl \
      TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=1,3 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-swiglu-dram-resident-768-wh-verify \
      "$PY" perf/ceilrfd3/wh_ladder.py >> $WT/perf/ceilrfd3/whverify/seed7.log 2>&1
  cp -f $WT/perf/ceilrfd3/whverify/seed7/cap$t.cif $WT/perf/ceilrfd3/whverify/seed7/keep_cap$t.cif 2>/dev/null
done
echo "[seed7] $(date -Is) DONE" >> $WT/perf/ceilrfd3/whverify/seed7.log
