#!/bin/bash
# After xmodel.sh: b32 as lanes of 16 on the branch, then 256 b4 on main and on the branch. Results in ~/.cache/l1d-runs.
R=/home/moritz/.cache/l1d-runs WT=/home/moritz/.coworker/wt/rfd4-l1decline
while pgrep -f "l1decline/(repro|xmodel)" >/dev/null || pgrep -f trimul_ab.py >/dev/null; do sleep 10; done
EXTRA="--lane-width 16" $WT/perf/rfd4/l1decline/repro.sh $R/branch-192b32l16 192 32
$WT/perf/rfd4/l1decline/repro.sh $R/main-256b4 256 4 /home/moritz/.cache/l1d-main
$WT/perf/rfd4/l1decline/repro.sh $R/branch-256b4 256 4
echo "chain2 done $(date -u +%FT%TZ)" > $R/chain2.done
