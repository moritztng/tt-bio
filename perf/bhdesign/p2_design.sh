#!/bin/bash
# p2: push the two design models past the fixture wall the parent pass hit.
#   $1 model (boltzgen|pxdesign)   $2 card (UMD id)   $3 comma-separated target-residue rungs
# One rung per subprocess (ladder.py does that), ascending, stopping at the first FAIL so the
# rung below it is the negative control rather than a rung nobody ran.
cd /home/ttuser/.coworker/wt/bh-1536-design-embed-p2 || exit 1
M=$1; CARD=$2; SIZES=$3
export TT_BIO_LEASE_TIMEOUT=2400
python3 perf/bhdesign/ladder.py \
  --model "$M" --sizes "$SIZES" \
  --target perf/bhdesign/targets/big_7324.cif \
  --card "$CARD" --holder "worker:bh-1536-design-embed-p2" \
  --work "perf/bhdesign/work_p2_$M" --timeout 3000 --stop-on-fail \
  --out "perf/bhdesign/p2/$M.jsonl"
echo "=== $M ladder done $(date -u +%H:%M:%SZ) ==="
