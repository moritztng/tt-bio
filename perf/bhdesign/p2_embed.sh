#!/bin/bash
# p2: walk the four embedding models that stopped at 8192 to esmc-300m parity (65536) or a wall.
# Top rung first: a pass there ends the model, a fail is bisected downward.
cd /home/ttuser/.coworker/wt/bh-1536-design-embed-p2 || exit 1
OUT=perf/bhdesign/p2
mkdir -p "$OUT"
export TT_BIO_LEASE_TIMEOUT=1800
LAD="python3 perf/bhdesign/ladder.py --card 2 --holder worker:bh-1536-design-embed-p2 --work perf/bhdesign/work_p2 --timeout 2400"
for m in saprot-35m esmc-600m saprot-650m saprot-1.3b; do
  echo "=== $m top rung 65536 $(date -u +%H:%M:%SZ) ==="
  $LAD --model "$m" --sizes 65536 --out "$OUT/$m.jsonl"
done
echo "=== embed top-rung pass done $(date -u +%H:%M:%SZ) ==="
