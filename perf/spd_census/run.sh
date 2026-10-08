#!/bin/bash
# spd-census: profiled folds of a tt-bio tree, one arm after another, under the chip's flock.
# usage: run.sh CHIP RUNDIR "NAME=ARM NAME=ARM ..." [INPUT]
# Layout under $B (default ~/spd/spd-census): tt-metal (Tracy build, v0.68.0 = the serving wheel; one build serves
# Wormhole and Blackhole), tree (tt-bio checkout; TREE overrides it), syslib (libhwloc.so.15 where the box lacks it).
# Box differences come from the environment: ENVSH (sourced first; the Galaxies' ~/japanfold/env.sh), PY (python with
# the serving deps), LOCK (the chip's flock file). An arm named "roof" runs roofline_wh.py instead of a fold.
set -u
CHIP=$1; RUN=$2; ARMS=$3; INPUT=${4:-c730}
B=${B:-$HOME/spd/spd-census}; ENVSH=${ENVSH:-$HOME/japanfold/env.sh}; PY=${PY:-python}
LOCK=${LOCK:-$HOME/spd/locks/chip$CHIP.lock}; TREE=${TREE:-$B/tree}
[ -f "$ENVSH" ] && . "$ENVSH" > /dev/null 2>&1
export TT_METAL_HOME=$B/tt-metal
export PYTHONPATH=$TREE:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib:$B/syslib
export TT_METAL_CACHE=$B/cache-$CHIP TT_METAL_LOGS_PATH=$B/tt-logs
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_DIR=$B/leases TT_BIO_LEASE_HOLDER=spd-census
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
       TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1
mkdir -p "$RUN" "$TT_BIO_LEASE_DIR" "$TT_METAL_CACHE"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$RUN/run.log"; }
exec 9> "$LOCK"
say "waiting for flock $LOCK"; flock -w ${WAIT:-1800} 9 || { say "flock busy"; exit 3; }
cd $TREE
say "start chip $CHIP head $(git rev-parse --short HEAD 2>/dev/null || cat REVISION 2>/dev/null) arms $ARMS input $INPUT"
for na in $ARMS; do
  N=${na%%=*}; A=${na#*=}
  say "arm $N ($A) start, load $(cut -d' ' -f1-3 /proc/loadavg)"
  if [ "$A" = roof ]; then
    timeout -s TERM 1900 timeout -s INT 1800 $PY perf/spd_census/roofline_wh.py "$RUN/roof.json" > "$RUN/$N.log" 2>&1
  else
    timeout -s TERM 11100 timeout -s INT 10800 $PY perf/spd_census/census.py --out "$RUN/$N" \
        --chip $CHIP --arm "$A" --input $INPUT > "$RUN/$N.log" 2>&1
  fi
  say "arm $N end rc=$?"
done
say "end"
