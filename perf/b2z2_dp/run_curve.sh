#!/bin/bash
# The whole measurement in one detached chain: the curve at the fixed cap, then the two cap
# probes at the widest width that say whether the cap is what bounds the wide arm.
set -u
WT=/home/mthuening/work/wt/b2z2-dp-throughput-linear
bash "$WT/perf/b2z2_dp/curve.sh" 8 "$WT/perf/b2z2_dp/curve" 32 16 8 4 2 1 1
bash "$WT/perf/b2z2_dp/curve.sh" 4 "$WT/perf/b2z2_dp/cap4w32" 32
bash "$WT/perf/b2z2_dp/curve.sh" 2 "$WT/perf/b2z2_dp/cap2w32" 32
echo "[$(date -u +%FT%TZ)] ALL DONE"
