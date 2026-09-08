#!/bin/bash
# The Wormhole A/B the pc pass could not run: 768 residues with the budget OFF, i.e. every
# residency granted, which is the unconditional-L1 path the branch replaced. Three attempts,
# because that path folds 768 only two times in three (state/ceiling-rfd3.md).
set -u
WT=/home/cust-team/mthuening/rfd3whv
PY=/home/cust-team/mthuening/tt-bio/env/bin/python
CARD=${1:-0}
cd "$WT" || exit 1
for i in 1 2 3; do
  env PYTHONPATH=$WT WH_TOTAL=768 WH_TAG=ab768nobudget WH_HOST_THREADS=30 \
      WH_OUT_DIR=$WT/perf/ceilrfd3/whverify/ab768_$i \
      WH_JSONL=$WT/perf/ceilrfd3/whverify/ab768.jsonl \
      TT_BIO_L1_RESIDENT_BUDGET_BYTES=0 \
      TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=1,$CARD \
      TT_BIO_LEASE_HOLDER=worker:rfd3-swiglu-dram-resident-768-wh-verify \
      "$PY" perf/ceilrfd3/wh_ladder.py >> $WT/perf/ceilrfd3/whverify/ab768.log 2>&1
  echo "[ab768] $(date -Is) attempt=$i rc=$?" >> $WT/perf/ceilrfd3/whverify/ab768.log
done
echo "[ab768] $(date -Is) DONE" >> $WT/perf/ceilrfd3/whverify/ab768.log
