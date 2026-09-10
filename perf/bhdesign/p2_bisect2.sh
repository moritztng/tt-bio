#!/bin/bash
# Give the other four embedding models their own bracket. saprot-35m was bisected from 31073 wide
# to 4096; the rest still publish 99999 because that is the largest size THEY measured, and a
# sibling's rung is not transferable -- the free-DRAM headroom differs with the resident weights,
# so each model's own L_max differs even though the failing allocation is identical.
#   $1 card (UMD id)   $2..  models
cd /home/ttuser/.coworker/wt/bh-1536-design-embed-p2 || exit 1
CARD=$1; shift
export TT_BIO_LEASE_TIMEOUT=2400
for m in "$@"; do
  echo "=== $m bisect on card $CARD $(date -u +%H:%M:%SZ) ==="
  python3 perf/bhdesign/ladder.py --model "$m" --sizes 114688,126976 \
    --card "$CARD" --holder worker:bh-1536-design-embed-p2 \
    --work "perf/bhdesign/work_p2_b$CARD" --timeout 2400 --stop-on-fail \
    --out "perf/bhdesign/p2/$m.jsonl"
done
echo "=== card $CARD bisect done $(date -u +%H:%M:%SZ) ==="
