#!/bin/bash
# The size sweep JapanFold actually serves, on one card under one lock.
#
# 640 and 1024 are not optional. 640 is the first size past SEQ_LEN_MORE_CHUNKING = 608, where
# Wormhole switches to the chunked triangle-attention and transition paths that Blackhole never
# takes below 1536, so everything above it is a Wormhole-only code path that has never been tuned
# on a 72-core, 12 GB part. 1024 is max_residues and the size where the trimul tail stops being
# attempted at all (0 served AND 0 declined, against 1120 declines at 512).
#
# One lock for the whole sweep: the curve is read across sizes, so the points cannot be separated
# by another worker's timed run. Warm median of 3 below 640, of 2 above, because a warm 1024 fold
# is ~250 s and this box is shared production.
#
# Usage: wh_sweep.sh [label-suffix]   (env: EXTRA="--flag ..." passed to every arm)
set -u
TREE=/home/cust-team/mthuening/whbase/tt-bio
OUT=/home/cust-team/mthuening/whbase/out/sweep
SUF=${1:-base}
EXTRA=${EXTRA:-}
. /home/cust-team/mthuening/whbase/pick_card.sh
CARD=$(pick_card) || { echo "no free card"; exit 70; }
echo "sweep $SUF on UMD $CARD $(date -u -Is)"
mkdir -p "$OUT"
cd "$TREE" || exit 1

for S in 128 256 298 320 384 512 640 768 1024; do
  REP=3; [ "$S" -ge 640 ] && REP=2
  echo "=== $S start $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 \
  HF_HUB_CACHE=/home/cust-team/mthuening/whbase/hfcache \
    ./env/bin/python perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" \
      --size "$S" --repeat "$REP" $EXTRA \
      --label "sweep_${SUF}_$S" --out "$OUT/${SUF}_$S.json" > "$OUT/${SUF}_$S.log" 2>&1
  echo "EXIT $S = $?"
  grep -h "median\|cold " "$OUT/${SUF}_$S.log" | tail -2
done
echo "SWEEP $SUF DONE $(date -u +%H:%M:%S)"
