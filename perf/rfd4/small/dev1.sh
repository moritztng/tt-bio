#!/bin/bash
# rfd4-perf-small device pass 1 on one qb2 chip: does trimul_t1_small (T1 + pair_tail from 128 tokens) pay?
#   dev1.sh <chip> <tree> <out>     tree = an exported checkout of wk/rfd4-perf-small
# (a) paired traced trunk passes, CAND vs CAND + trimul_t1_small, at 128 and 192 (192 is the bucket the lever also
#     reroutes from L1 to DRAM); (b) the trimul alone per call with float64 error; (c) census v3 at 128 (Tracy),
#     CAND and CAND + trimul_t1_small. AICLK sampled inside every python leg; load1 stamped per leg.
set -u
C=$1; T=$2; R=$3
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail
mkdir -p $R
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 10800 9 || { stamp "no lock"; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-small PYTHONPATH=$T:$T/perf
cd $T
stamp start
for n in 128 192; do
  timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $W --out $R/stack-$n.json --n $n --reps 30 \
    --base $CAND --levers $CAND,trimul_t1_small > $R/stack-$n.log 2>&1
  stamp "stack-$n rc=$?"
done
for n in 128 192; do
  timeout 1200 $PY perf/rfd4/tri2/bench.py --weights $W --out $R/tri-$n.json --n $n --reps 30 \
    --arms "lever:${CAND//,/+};mask:none|lever:${CAND//,/+}+trimul_t1_small;mask:none" \
    > $R/tri-$n.log 2>&1
  stamp "tri-$n rc=$?"
done
for arm in cand small; do
  L=$CAND; [ $arm = small ] && L=$CAND,trimul_t1_small
  TT_BIO_LEVERS=$L bash perf/rfd4/e2e/elt_prof.sh $R/census-$arm $C --n 128 > $R/census-$arm.log 2>&1
  stamp "census-$arm rc=$?"
done
stamp ALLDONE
