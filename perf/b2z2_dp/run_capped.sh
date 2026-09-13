#!/bin/bash
# Wait out the uncapped control arm, then run the capped ladder. Same pool, same reps, and a
# second width-1 arm at the end so the A/A floor is the same estimator as the ratios.
set -u
WT=/home/mthuening/work/wt/b2z2-dp-throughput-linear
while pgrep -f "dp_width.py --cards 0,1,2,3,4,5,6,7 " > /dev/null; do sleep 15; done
sleep 20
exec bash "$WT/perf/b2z2_dp/ladder.sh" \
  "0,1,2,3,4,5,6,7,12,13,14,15,17,24,25,26" "1,2,4,8,16,1" 3 "$WT/perf/b2z2_dp/capped"
