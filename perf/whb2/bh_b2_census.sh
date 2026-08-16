#!/bin/bash
# Blackhole counterpart of the Wormhole Boltz-2 gate census: same harness, same fixture, card 1.
# Answers one question -- is the F1 trimul-tail decline a Wormhole effect or the shape of the model.
set -u
WT=/home/ttuser/.coworker/wt/wh-perf-boltz2
OUT=$WT/perf/whb2
mkdir -p "$OUT"
cd "$WT" || exit 1
TT_VISIBLE_DEVICES=1 TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 \
  python3 perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$WT" --size 512 --repeat 1 \
    --label bhcensus_512_qb1c1 --out "$OUT/b2_census_512_qb1c1.json" \
    > "$OUT/b2_census_512_qb1c1.log" 2>&1
echo "EXIT $?" >> "$OUT/b2_census_512_qb1c1.log"
