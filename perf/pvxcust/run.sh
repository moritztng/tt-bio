#!/usr/bin/env bash
# One measured session: take the box, pin the clock inside the lock, fold, release both.
#   run.sh <tag> <legs>     legs = comma-separated SIZE:NSAMPLE:NWARM
# The pin starts INSIDE the lock: a pin taken before the lock would hold the card at 1350 MHz
# for the whole queue wait, heating the part that is about to be measured.
set -u
TAG="${1:?tag}"; LEGS="${2:?legs}"
WT=/home/ttuser/.coworker/wt/pvx-custchart
cd "$WT" || exit 1
export PVXC_TAG="$TAG" PVXC_LEGS="$LEGS"
/home/ttuser/.coworker/scripts/benchlock.sh "pvx-custchart/$TAG" -- \
  bash perf/pvxcust/inner.sh > "$WT/perf/pvxcust/${TAG}.outer.log" 2>&1
echo "rc=$?" >> "$WT/perf/pvxcust/${TAG}.outer.log"
