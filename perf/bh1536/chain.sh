#!/bin/bash
# The 1536 bar first, for every model in the registry. A model that clears it needs no ladder;
# a model that does not gets one walked downward in a later pass.
WT=/home/ttuser/.coworker/wt/bh-1536-structure
cd $WT || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export PATH=/home/ttuser/.tenstorrent-venv/bin:$PATH
JL=$WT/perf/bh1536/results.jsonl
for spec in "$@"; do
  model=${spec%%:*}; size=${spec##*:}
  if [ -f "$JL" ] && grep -q "\"model\": \"$model\", \"size\": $size, \"tag\": \"\"" "$JL"; then
    echo "skip $model $size (already recorded)"; continue
  fi
  echo "=== $(date -u +%FT%TZ) $model $size ==="
  $PY perf/bh1536/run_rung.py --model "$model" --size "$size" --budget 2700
done
echo "CHAIN DONE $(date -u +%FT%TZ)"
