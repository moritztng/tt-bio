#!/bin/bash
# The whole measurement in one detached chain: the curve at the fixed cap, then the cap probes.
#
# The probes are at width 16, not 32. Device open is serialized host-wide, about 50 s a chip, so a
# 32-way arm spends ~25 minutes bringing chips up before it measures anything; 16 already answers
# whether the wide arm is bound by the host thread cap, at a third of the cost.
set -u
WT=/home/mthuening/work/wt/b2z2-dp-throughput-linear
bash "$WT/perf/b2z2_dp/curve.sh" 8 "$WT/perf/b2z2_dp/curve" 32 16 8 4 2 1 1
bash "$WT/perf/b2z2_dp/curve.sh" 4 "$WT/perf/b2z2_dp/cap4" 16
bash "$WT/perf/b2z2_dp/curve.sh" 2 "$WT/perf/b2z2_dp/cap2" 16
echo "[$(date -u +%FT%TZ)] ALL DONE"
