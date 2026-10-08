#!/bin/bash
# spd-census on a Wormhole Galaxy chip: profiled folds of the staging tree, one arm after another, under the
# chip's flock. usage: run_wh.sh CHIP RUNDIR "NAME=ARM NAME=ARM ..." [INPUT]
# Layout under ~/spd/spd-census: tt-metal (Tracy build, v0.68.0 = the serving wheel), tree (tt-bio checkout).
set -u
CHIP=$1; RUN=$2; ARMS=$3; INPUT=${4:-c730}
B=$HOME/spd/spd-census
. ~/japanfold/env.sh > /dev/null 2>&1
export TT_METAL_HOME=$B/tt-metal
export PYTHONPATH=$B/tree:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib
export TT_METAL_CACHE=$B/cache-$CHIP TT_METAL_LOGS_PATH=$B/tt-logs
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_DIR=$B/leases TT_BIO_LEASE_HOLDER=spd-census
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
       TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1
mkdir -p "$RUN" "$TT_BIO_LEASE_DIR" "$TT_METAL_CACHE"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$RUN/run.log"; }
exec 9> ~/spd/locks/chip$CHIP.lock
say "waiting for flock chip $CHIP"; flock -w 1800 9 || { say "flock busy"; exit 3; }
cd $B/tree
say "start chip $CHIP head $(git rev-parse --short HEAD) arms $ARMS input $INPUT"
for na in $ARMS; do
  N=${na%%=*}; A=${na#*=}
  say "arm $N ($A) start"
  timeout -s TERM 11100 timeout -s INT 10800 nice -n 0 python perf/spd_census/census.py --out "$RUN/$N" \
      --chip $CHIP --arm "$A" --input $INPUT > "$RUN/$N.log" 2>&1
  say "arm $N end rc=$?"
done
say "end"
