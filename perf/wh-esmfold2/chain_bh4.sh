#!/bin/bash
# wh-perf-esmfold2 p6: the Blackhole neutrality A/B, post-fix, on card 2.
# The p5 legs were pointed at card 1, which is not this task's assignment and is now legitimately
# held by wh-perf-boltz2 -- the non-fast leg died on the device lease with DeviceInUseError after
# waiting out 909 s of benchlock first. Card 2 is this task's card.
# Arm A on a 13x10 grid sets a flag the shipped expression must never read, so both legs are
# deliberate A/A runs and the only acceptable result is no difference.
cd /home/ttuser/.coworker/wt/wh-perf-esmfold2 || exit 1
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_HOLDER=worker:wh-perf-esmfold2
export TT_METAL_LOGGER_LEVEL=FATAL
BL=/home/ttuser/.coworker/scripts/benchlock.sh
O=perf/wh-esmfold2/out
for spec in "fast --fast" "plain "; do
  set -- $spec; tag=$1; shift
  echo "=== bh A/A $tag start $(date -u +%H:%M:%S) ==="
  $BL wh-perf-esmfold2 -- python3 -u perf/wh-esmfold2/fold_ab512.py \
      --model esmfold2 --size 512 $@ --arms base,A --rounds 2 \
      --out "$O/fold_ab_512_qb1c2_$tag.json" > "$O/fold_ab_512_qb1c2_$tag.log" 2>&1
  echo "=== bh A/A $tag rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== chain_bh4 done $(date -u +%H:%M:%S) ==="
