#!/bin/bash
# p2, second embedding pass: every model that cleared 65536 gets walked UP until it breaks.
# Parity with esmc-300m was the bar; a real wall is what CEILINGS can be written from, and the
# top rung costs 78-111 s, so the doubling is cheap. --stop-on-fail leaves the rung below the
# wall passed and on record, which is the negative control the ceiling row needs.
cd /home/ttuser/.coworker/wt/bh-1536-design-embed-p2 || exit 1
OUT=perf/bhdesign/p2
export TT_BIO_LEASE_TIMEOUT=1800
LAD="python3 perf/bhdesign/ladder.py --card 2 --holder worker:bh-1536-design-embed-p2 --work perf/bhdesign/work_p2 --timeout 2400 --stop-on-fail"
# 99999 is deliberately NOT a multiple of 32 (3124.97 x 32): the row-count and nonzero-fraction
# check can only tell a correct implementation from one that silently rounds up at a size that
# is off the bucket. Every other rung here is a power of two and cannot.
for m in saprot-35m esmc-600m saprot-650m saprot-1.3b esmc-300m; do
  echo "=== $m walk up $(date -u +%H:%M:%SZ) ==="
  $LAD --model "$m" --sizes 99999,131072,262144,524288 --out "$OUT/$m.jsonl"
done
echo "=== embed walk-up done $(date -u +%H:%M:%SZ) ==="
