#!/bin/bash
# The timed ladder re-taken on a quiet box.
#
# Every s/round in the first three ladders was taken with two to four of my own legs on other
# cards of this box, at load1 5 to 12. A BindCraft 2 round is a host column and a device column
# laid end to end, so a job on another card lengthens it: 320 tokens read 16.45 s at load1 5.4
# against 288's 10.16 s at load1 2.5, 1.6x for one token bucket, with AICLK flat at 1343-1350.
# That is contention, not the size. This leg runs the whole ladder sequentially on one card with
# nothing of mine beside it but the two full campaigns, and six rounds a rung instead of four so
# the median has something to sit on.
#
#   ladder_quiet.sh <card> <spec>...   spec = "<target> <binder> <tokens>"
set -u
cd "$(dirname "$0")/../.."
card=$1; tag=$2; shift 2
log=perf/bgx_size/out/$tag.log
mkdir -p perf/bgx_size/out
: > "$log"
for spec in "$@"; do
    set -- $spec
    echo "=== $(date -u +%FT%TZ) $tag rung $1/$2 -> $3 tokens ===" >> "$log"
    timeout 5400 perf/bgx_size/run_rung.sh "$card" "q$3_$1_$2" \
        --target "$1" --binder "$2" --rounds 6 --trajectories 1 >> "$log" 2>&1
    echo "=== rc=$? ===" >> "$log"
done
echo "=== $tag DONE $(date -u +%FT%TZ) ===" >> "$log"
