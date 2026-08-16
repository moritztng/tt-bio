#!/bin/bash
# wh-perf-esmfold2 p4: the Blackhole neutrality A/B. On qb1's 13x10 grid _IS_SMALL_GRID is False,
# so arm `A` sets a flag that the shipped expression never reads -- this is a deliberate A/A, and
# a delta outside its own spread would mean the neutrality argument of section 7 is wrong.
cd /home/ttuser/.coworker/wt/wh-perf-esmfold2 || exit 1
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:wh-perf-esmfold2
export TT_METAL_LOGGER_LEVEL=FATAL
BL=/home/ttuser/.coworker/scripts/benchlock.sh
O=perf/wh-esmfold2/out
for spec in "fast --fast" "plain "; do
  set -- $spec; tag=$1; shift
  echo "=== bh neutrality $tag start $(date -u +%H:%M:%S) ==="
  $BL wh-perf-esmfold2 -- python3 -u perf/wh-esmfold2/fold_ab512.py \
      --model esmfold2 --size 512 $@ --arms base,A --rounds 2 \
      --out "$O/fold_ab_512_qb1c1_$tag.json" > "$O/fold_ab_512_qb1c1_$tag.log" 2>&1
  echo "=== bh neutrality $tag rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== chain_bh2 done $(date -u +%H:%M:%S) ==="
