#!/usr/bin/env bash
# One census run: take the box, pin the clock, fold, release the pin, release the box.
#   run.sh <model> <tag> [extra census.py args...]
# Card 3 (UMD) is node 0 on qb1; PVX_CARD / PVX_CLK_NODE move both together.
#
# The pin starts INSIDE the lock. Three pvx rows share `benchlock`, so a pin taken before
# the lock holds the card at 1350 MHz for the whole queue wait -- burning 55 W and heating
# the part that is about to be measured, on a box where the watchdog margin already scales
# with load.
set -u
MODEL="${1:?model}"; TAG="${2:?tag}"; shift 2
WT=/home/ttuser/.coworker/wt/pvx-protenix-specific
cd "$WT" || exit 1
export PVX_MODEL="$MODEL" PVX_TAG="$TAG" PVX_ARGS="$*"
/home/ttuser/.coworker/scripts/benchlock.sh "pvx-protenix-specific/$TAG" -- \
  bash perf/pvx_ps/inner.sh > "$WT/perf/pvx_ps/${TAG}.outer.log" 2>&1
echo "rc=$?" >> "$WT/perf/pvx_ps/${TAG}.outer.log"
