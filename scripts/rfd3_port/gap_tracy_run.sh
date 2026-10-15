#!/bin/bash
# gap_tracy_run.sh CHIP OUT TIMEOUT_S [gap_tracy.py args...]: one RFD3 device census on one chip, through the Tracy
# build of tt-metal, under the chip's flock, with every TT node's AICLK and the load sampled every 0.5 s DURING the run.
#   OUT/census.json  the census (gap_tracy.py), OUT/harness.log its stdout, OUT/aiclk.tsv, OUT/run.log
# B = the census home with the Tracy build (default ~/spd/spd-census, tt-metal v0.68.0 = the serving wheel, one build
# serves Wormhole and Blackhole). PY = interpreter with the serving deps (default python, after a Galaxy's env.sh).
# LOCK = the chip's flock file (default ~/spd/locks/chip<CHIP>.lock), WAIT = seconds to queue on it.
set -u
CHIP=$1 OUT=$2 TMO=$3; shift 3
B=${B:-$HOME/spd/spd-census}
[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
cd "$(dirname "$0")/../.."
mkdir -p "$OUT" ~/spd/locks ~/spd/spd-gap-whrfd3/leases
TM=$B/tt-metal
export TT_METAL_HOME=$TM PYTHONPATH=$PWD:$TM/ttnn:$TM/tools:$TM LD_LIBRARY_PATH=$TM/build_Release/lib:$B/syslib
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=spd-gap-whrfd3 TT_BIO_LEASE_DIR=$HOME/spd/spd-gap-whrfd3/leases
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
       TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
# a private profiler dir and kernel cache: the zone log then holds only this run's kernel paths (no stale hash clash)
export TT_METAL_PROFILER_DIR=$OUT/profiler TT_METAL_CACHE=$OUT/cache TT_METAL_LOGS_PATH=$OUT/tt-logs
rm -rf "$OUT/profiler" "$OUT/cache"
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
exec 9> "${LOCK:-$HOME/spd/locks/chip$CHIP.lock}"
flock -w "${WAIT:-600}" 9 || { say "flock busy"; exit 3; }
say "start chip $CHIP tree $(cat TREE_SHA 2>/dev/null || git rev-parse --short HEAD) host $(hostname) args: $*"
( while :; do l=$(cut -d' ' -f1 /proc/loadavg); for n in /sys/class/tenstorrent/tenstorrent\!*; do
    printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done; sleep 0.5; done ) 9>&- >> "$OUT/aiclk.tsv" 2>/dev/null &
SAMP=$!
timeout -s TERM $((TMO + 120)) timeout -s INT "$TMO" "${PY:-python}" scripts/rfd3_port/gap_tracy.py "$OUT/census.part.json" "$@" \
  > "$OUT/harness.log" 2>&1 &
H=$!
# the device node the run opened (OUT/nodes.txt) names which aiclk.tsv column is this chip
( while kill -0 $H 2>/dev/null; do sleep 5
    for p in $(pstree -p $H | grep -o "([0-9]*)" | tr -d "()"); do ls -l /proc/$p/fd 2>/dev/null | grep -o 'tenstorrent/[0-9]*'; done | sort -u > "$OUT/nodes.tmp"
    [ -s "$OUT/nodes.tmp" ] && { mv "$OUT/nodes.tmp" "$OUT/nodes.txt"; break; }
  done; rm -f "$OUT/nodes.tmp" ) 9>&- &
wait $H; rc=$?
kill $SAMP 2>/dev/null
[ $rc = 0 ] && mv "$OUT/census.part.json" "$OUT/census.json"
say "end rc=$rc"
exit $rc
