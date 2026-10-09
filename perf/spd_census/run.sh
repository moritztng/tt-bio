#!/bin/bash
# spd-census: profiled folds of a tt-bio tree, one arm after another, under the chip's flock.
# usage: run.sh CHIP RUNDIR "NAME=ARM NAME=ARM ..." [INPUT]
# Layout under $B (default ~/spd/spd-census): tt-metal (Tracy build, v0.68.0 = the serving wheel; one build serves
# Wormhole and Blackhole), tree (tt-bio checkout; TREE overrides it), syslib (libhwloc.so.15 where the box lacks it).
# Box differences come from the environment: ENVSH (sourced first; the Galaxies' ~/japanfold/env.sh), PY (python with
# the serving deps), LOCK (the chip's flock file), MODEL (census.py --model; Protenix-v2 when unset). An arm named "roof" runs roofline_wh.py instead of a fold.
set -u
CHIP=$1; RUN=$2; ARMS=$3; INPUT=${4:-c730}
B=${B:-$HOME/spd/spd-census}; ENVSH=${ENVSH:-$HOME/japanfold/env.sh}; PY=${PY:-python}
LOCK=${LOCK:-$HOME/spd/locks/chip$CHIP.lock}; TREE=${TREE:-$B/tree}
[ -f "$ENVSH" ] && . "$ENVSH" > /dev/null 2>&1
export TT_METAL_HOME=$B/tt-metal
export PYTHONPATH=$TREE:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib:$B/syslib
export TT_METAL_LOGS_PATH=$B/tt-logs
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_DIR=$B/leases TT_BIO_LEASE_HOLDER=spd-census
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
       TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1
# Without a Tracy server attached, every device marker pushed to Tracy stays queued in this process: r4b grew to
# 257 GB RSS in one c730 fold and was OOM-killed. census.py reads the analyses, not Tracy, so do not push.
export TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
# The profiler keys device zones by a 16-bit hash of "name,file,line" and throws on a collision. Its zone log
# accumulates every run's kernel paths, so a stale entry from an earlier tree can collide with this one (r4: two trees'
# triatt compute_streaming.hpp; r9: staging6's ttnn SDPA kernel against staging9's metal-overlay copy). Give each run
# its own profiler dir and kernel cache: every kernel then compiles here and the log holds only this run's zones.
# A tree with a metal overlay (staging9's silu_f32) still compiles some ttnn headers under two paths, the overlay's
# and the stock one, and those can collide by chance (~1 in 10 per layout). The overlay lives under XDG_CACHE_HOME
# (env.sh sets HF_HOME and TT_BIO_CACHE explicitly, so moving it moves nothing else); a colliding arm reruns (up to
# twice) with the overlay at a different path, which rerolls every overlay hash.
export TT_METAL_PROFILER_DIR=$RUN/profiler TT_METAL_CACHE=$RUN/cache
mkdir -p "$RUN" "$TT_BIO_LEASE_DIR"
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
    rc=$?
  else
    for try in 0 1 2; do
      export XDG_CACHE_HOME=$RUN/xdg$try
      rm -rf "$TT_METAL_PROFILER_DIR" "$TT_METAL_CACHE" "$RUN"/xdg*; mkdir -p "$TT_METAL_CACHE"
      timeout -s TERM 11100 timeout -s INT 10800 $PY perf/spd_census/census.py --out "$RUN/$N" \
          --chip $CHIP --arm "$A" --input $INPUT ${MODEL:+--model $MODEL} > "$RUN/$N.log" 2>&1
      rc=$?
      grep -q "hashes are colliding" "$RUN/$N.log" || break
      say "arm $N try $try: profiler zone-hash collision, rerunning with the overlay elsewhere"
      mv "$RUN/$N.log" "$RUN/$N.collision$try.log"; rm -rf "$RUN/$N"
    done
  fi
  say "arm $N end rc=$rc"
done
rm -rf "$TT_METAL_CACHE" "$RUN"/xdg*
say "end"
