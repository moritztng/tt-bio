#!/bin/bash
# gap_tracy_run.sh CHIP OUT TIMEOUT_S [gap_tracy.py args...]: one RFD3 device census on one chip, through the Tracy
# build of tt-metal, via gap_run.sh (chip flock, AICLK sampled during the run). OUT/census.json is the census.
# B = the census home with the Tracy build (default ~/spd/spd-census, tt-metal v0.68.0 = the serving wheel, one build
# serves Wormhole and Blackhole).
set -u
CHIP=$1 OUT=$2 TMO=$3; shift 3
B=${B:-$HOME/spd/spd-census}
TM=$B/tt-metal
export TT_METAL_HOME=$TM PYTHONPATH=$TM/ttnn:$TM/tools:$TM LD_LIBRARY_PATH=$TM/build_Release/lib:$B/syslib
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
       TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
# a private profiler dir and kernel cache: the zone log then holds only this run's kernel paths (no stale hash clash)
export TT_METAL_PROFILER_DIR=$OUT/profiler TT_METAL_CACHE=$OUT/cache TT_METAL_LOGS_PATH=$OUT/tt-logs
rm -rf "$OUT/profiler" "$OUT/cache"
bash "$(dirname "$0")/gap_run.sh" "$CHIP" "$OUT" "$TMO" '"$PY"' scripts/rfd3_port/gap_tracy.py "$OUT/census.part.json" "$@" \
  && mv "$OUT/census.part.json" "$OUT/census.json"
