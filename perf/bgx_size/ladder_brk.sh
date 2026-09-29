#!/bin/bash
# Bracket the 512-token pothole.  ladder_brk.sh <card> [rounds]
#
# 544 holds 14.2255 GB and runs 48.95 s a round; 512 holds 25.7543 and runs 68.48. Same
# target (hIL2R), same two chains, same everything but the token axis -- the LARGER size is
# 1.8x lighter and 1.4x faster. At 512 the fused triangle attention is refused L1 and
# declines to the composed path, which holds the whole [N,4,N,N]; at 544 it serves.
#
# So 512 is a pothole, not a ceiling, and these rungs say how wide it is. 256 is here
# because 512 is a power of two and the chunking ladder's block choice is what decides the
# circular-buffer size: if the gate is really "power of two", 256 potholes too and that is
# a size researchers use every day. 512 is repeated on a different target with a different
# chain count to show the cliff follows the token axis and not the input.
set -u
cd "$(dirname "$0")/../.."
card=${1:-0}
rounds=${2:-4}
foot=${3:-}
log=perf/bgx_size/out/ladder_brk_card$card$foot.log
mkdir -p perf/bgx_size/out
: > "$log"
run() {   # run <target> <binder> <tokens> <tag-suffix>
    echo "=== $(date -u +%FT%TZ) bracket rung $1/$2 -> $3 tokens ===" >> "$log"
    timeout 5400 perf/bgx_size/run_rung.sh "$card" "${foot:-t}$3_$1_$2" \
        --target "$1" --binder "$2" --rounds "$rounds" --trajectories 1 \
        ${foot:+--footprint} >> "$log" 2>&1
    echo "=== rc=$? ===" >> "$log"
}
run hPDL1  141 256
run hCA2   150 416
run hIL2R  50  448
run hIL2R  90  480
run hTNFa  50  512
echo "=== BRACKET DONE $(date -u +%FT%TZ) ===" >> "$log"
