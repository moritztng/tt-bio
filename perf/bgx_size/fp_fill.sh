#!/bin/bash
# The footprint rungs the first two ladders left blank, all with the lever counters.
#   fp_fill.sh <card> <spec>...    spec = "<target> <binder> <tokens>"
set -u
cd "$(dirname "$0")/../.."
card=$1; tag=$2; shift 2
log=perf/bgx_size/out/$tag.log
mkdir -p perf/bgx_size/out
: > "$log"
for spec in "$@"; do
    set -- $spec
    echo "=== $(date -u +%FT%TZ) $tag rung $1/$2 -> $3 tokens ===" >> "$log"
    timeout 5400 perf/bgx_size/run_rung.sh "$card" "fp$3_$1_$2" \
        --target "$1" --binder "$2" --rounds 3 --trajectories 1 --footprint >> "$log" 2>&1
    echo "=== rc=$? ===" >> "$log"
done
echo "=== $tag DONE $(date -u +%FT%TZ) ===" >> "$log"
