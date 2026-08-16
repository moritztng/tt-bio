#!/usr/bin/env bash
# One process per candidate under an OS timeout. A Python SIGALRM handler cannot preempt a ttnn
# device call, so the in-process backstop the first version used never fired and the sweep stalled
# on its first block.
WT=/home/ttuser/.coworker/wt/protenix-v2-beat-dgx-h200
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-beat-dgx-h200 PYTHONPATH="$WT"
for C in 4,8,2,4,2 4,8,4,4,2 4,8,16,4,2 16,8,8,4,2 8,8,16,4,2; do
  echo "=== BLOCK $C  $(date -Is) ==="
  timeout 90 $PY -u perf/pxbeat/trans_block_sweep.py --only "$C" --reps 3 \
      --out "perf/pxbeat/block_${C//,/_}.json" 2>&1 \
    | grep -v "Config{\|DEBUG\|^$"
  echo "  exit=$?"
done
echo "=== SWEEP DONE $(date -Is) ==="
