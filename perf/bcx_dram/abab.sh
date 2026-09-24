#!/bin/bash
# Interleaved step time: the fixed tree (A) against the pre-fix tree with a collect after
# every step (B), alternated so load drift on qb2 lands on both arms. 8 steps each at n=307.
set -u
WT=/home/ttuser/.coworker/wt/bcx-dram
PRE=/home/ttuser/bcx-dram-prefix
OUT=$WT/perf/bcx_dram/abab
mkdir -p $OUT
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-dram
PY=/home/ttuser/bcx_e2e_venv/bin/python
for rep in 1 2; do
  for arm in A B; do
    if [ $arm = A ]; then cd $WT; extra=""; else cd $PRE; extra="--gc-probe"; fi
    $PY perf/bcx_dram/memtrace.py --seed 200 --trajectory 1 --steps 8 --card 2 $extra \
      --out $OUT/${arm}${rep}.json > $OUT/${arm}${rep}.log 2>&1 < /dev/null
    echo "$arm$rep exit $? $(date -u +%FT%TZ)" >> $OUT/progress.txt
  done
done
echo done >> $OUT/progress.txt
