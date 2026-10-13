#!/bin/bash
# Profiled ESMFold2 folds under a qb1 card's flock, spd-census's Tracy environment (perf/spd_census/run.sh on
# wk/spd-census, whose comments say why each variable is set).
# usage: census.sh CARD RUNDIR "NAME=MODEL[:fast] ..."     e.g. census.sh 2 ~/spd/spd-gap-esmf/c1 "ef2=esmfold2 ef2f=esmfold2-fast"
set -u
CHIP=$1; RUN=$2; ARMS=$3
B=${B:-$HOME/spd/spd-census}; PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
TREE=$(cd "$(dirname "$0")/../.." && pwd); LOCK=${LOCK:-$HOME/spd_qb1_card$CHIP.lock}
export TT_METAL_HOME=$B/tt-metal
export PYTHONPATH=$TREE:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib:$B/syslib
export TT_METAL_LOGS_PATH=$RUN/tt-logs TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-spd-gap-esmf}
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
       TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
# Fresh profiler dir and kernel cache per run (zone-hash collisions, see run.sh). XDG_CACHE_HOME stays: it
# holds the HF hub cache, and moving it re-downloads ESMC-6B.
export TT_METAL_PROFILER_DIR=$RUN/profiler TT_METAL_CACHE=$RUN/cache
mkdir -p "$RUN"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$RUN/run.log"; }
G="release_gate.py|gate_fanout.py|perf_ab"
while pgrep -f "$G" >/dev/null; do sleep 60; done
exec 9> "$LOCK"; say "waiting for flock $LOCK"; flock -w ${WAIT:-43200} 9 || { say "flock busy"; exit 3; }
cd "$TREE"; say "start card $CHIP tree $(cat TREE_SHA 2>/dev/null) arms $ARMS"
for na in $ARMS; do
  N=${na%%=*}; A=${na#*=}; M=${A%%:*}; FAST=; [ "${A#*:}" = fast ] && FAST=--fast
  rm -rf "$TT_METAL_PROFILER_DIR" "$TT_METAL_CACHE"; mkdir -p "$TT_METAL_CACHE"
  say "arm $N ($A) start, load $(cut -d' ' -f1-3 /proc/loadavg)"
  ( while :; do for n in /sys/class/tenstorrent/tenstorrent\!*; do printf '%s\t%s\t%s\n' "$(date +%s)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)"; done; sleep 1; done ) >> "$RUN/$N.aiclk.tsv" 2>/dev/null &
  S=$!
  timeout -s TERM 7300 timeout -s INT 7200 $PY perf/spd_gap_esmf/census_esm.py --out "$RUN/$N" --model $M $FAST > "$RUN/$N.log" 2>&1
  rc=$?; kill $S 2>/dev/null
  say "arm $N end rc=$rc"
done
rm -rf "$TT_METAL_CACHE"
say "end"
