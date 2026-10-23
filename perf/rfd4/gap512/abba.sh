#!/bin/bash
# rfd4-perf-gap512 r3: A vs P in ABBA ABBA order (a clock that drifts through the session cancels), 4 designs a leg,
# 16 a side, at 512 then 384, one chip one session. Each leg records its AICLK samples (timeline.json "aiclk").
#   A=<sha> P=<sha> bash abba.sh <chip> <outdir> [wait_for_done_file]
set -u
N=$1; R=$2; WAIT=${3:-}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $R; R=$(cd $R && pwd)
[ -n "$WAIT" ] && until grep -q ALLDONE $WAIT 2>/dev/null; do sleep 30; done
LEGS="sA1 sP1 sP2 sA2 sA3 sP3 sP4 sA4"
DESIGNS=4 SIZE=512 HOLD_S=3000 bash $HERE/run.sh $N $R/n512 "$LEGS"
DESIGNS=4 SIZE=384 HOLD_S=2400 bash $HERE/run.sh $N $R/n384 "$LEGS"
echo "ALLDONE $(date -u +%FT%TZ)" >> $R/done
