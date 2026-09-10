#!/bin/bash
# Regenerate the campaign evidence log from the rung records and drop it where the DONE_CHECK
# reads. sweep.log is GENERATED; nothing here is hand-kept.
set -e
WT=/home/ttuser/.coworker/wt/bh-1536-design-embed-p2
S=/home/ttuser/.coworker/state/bh-1536-design-embed-p2
mkdir -p "$S"
cd "$WT"
python3 perf/bhdesign/render_sweep.py --dir perf/bhdesign/p2 > "$S/sweep.log"
wc -c "$S"/*.md "$S"/*.log 2>/dev/null || true
