#!/bin/bash
# Boltz-2 gate census on the Wormhole Galaxy: which fused kernels serve a call at three sizes.
# Not a timed protocol (three folds share the host), so the walls it prints are sanity checks,
# not results. One free card each, UMD 27/28/29 = /dev/tenstorrent/3,4,5.
set -u
TREE=/home/cust-team/mthuening/whbase/tt-bio
OUT=/home/cust-team/mthuening/whbase/out/census
mkdir -p "$OUT"
cd "$TREE" || exit 1

run() {  # size card repeat_or_census
  local size=$1 card=$2 mode=$3
  local extra="--repeat $mode"
  [ "$mode" = "census" ] && extra="--census"
  TT_VISIBLE_DEVICES=$card TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 \
    ./env/bin/python perf/of3_4xpd/xmodel_ab.py \
      --model boltz2 --tree "$TREE" --size "$size" $extra \
      --label "whcensus_$size" --out "$OUT/b2_$size.json" \
      > "$OUT/b2_$size.log" 2>&1
  echo "EXIT $size = $?" >> "$OUT/b2_$size.log"
}

run 384 27 1 &
run 512 28 1 &
run 1024 29 census &
wait
echo ALL_DONE
