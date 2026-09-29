#!/bin/bash
# The footprint leg of the size ladder: the same rungs with tenstorrent.dram_peak armed.
#
# Separate from the timed leg because the probe drains the pipeline. These runs report
# resident peak and free at the peak; their round times are meaningless and the harness
# marks them `timing_valid: false`.
#
# Three rounds rather than four: the high-water mark is reached inside the backward of
# every round, and a fourth adds cost without adding a number.
set -u
cd "$(dirname "$0")/../.."
card=${1:-3}
log=perf/bgx_size/out/ladder_fp_card$card.log
mkdir -p perf/bgx_size/out
: > "$log"
run() {   # run <target> <binder> <tokens>
    echo "=== $(date -u +%FT%TZ) footprint rung $1/$2 -> $3 tokens ===" >> "$log"
    timeout 3600 perf/bgx_size/run_rung.sh "$card" "fp$3_$1_$2" \
        --target "$1" --binder "$2" --rounds 3 --trajectories 1 --footprint >> "$log" 2>&1
    echo "=== rc=$? ===" >> "$log"
}
run hPDL1  50  192
run hPDL1  146 288
run hIL7RA 100 320
run hCA2   100 384
run hIL2R  100 512
run hIL2R  146 544
run hTNFa  100 576
run hHSA   100 704
echo "=== FOOTPRINT LADDER DONE $(date -u +%FT%TZ) ===" >> "$log"
