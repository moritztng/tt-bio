#!/usr/bin/env bash
# Wait for the 480 rung to exit, then run the 384 control on the same card.
#
# 384 is the rung of3t-crop640 measured on qb2 (17,924,142,080 B), so running it here is what
# ties this board's ladder to that one. `kill -0` on an explicit pid, not a pgrep pattern: a
# pattern match would find this wrapper's own argv.
set -u
PID=$1
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
exec bash perf/of3t_crop768/run_rung_nolock.sh 384 1
