#!/bin/bash
# The ladder above 704 tokens.  ladder_top.sh <card> [rounds]
#
# The first ladder stopped at 704 because bcx-bigtarget put the ceiling at 512 (544 with
# the fp32-softmax block) and 704 was already well past it. On the SHIPPED path it is not:
# bcx-bigtarget measured `--arm stack`, which its own harness defines as the levers off,
# and the shipped BindCraft 2 round arms `tri_att_sdpa_hifi`, whose backward recomputes one
# score block per chunk instead of holding the whole [N,4,N,N]. That term is 96*N^3 bytes
# and was 41-49 % of the footprint bigtarget measured. Measured at 320 tokens: 5.3383 GB on
# the shipped path against bigtarget's 8.6447, and 8.6447 - 96e-9*320^3 = 5.499.
#
# So the ceiling has to be found rather than assumed, and these are the rungs above the
# first ladder's top. Every rung is --trajectories 1 explicitly (bgx-orchestrator,
# 2026-09-29) so it measures the size and not the auto-interleave default.
set -u
cd "$(dirname "$0")/../.."
card=${1:-1}
rounds=${2:-4}
foot=${3:-}
log=perf/bgx_size/out/ladder_top_card$card$foot.log
mkdir -p perf/bgx_size/out
: > "$log"
run() {   # run <target> <binder> <tokens>
    echo "=== $(date -u +%FT%TZ) top rung $1/$2 -> $3 tokens ===" >> "$log"
    timeout 5400 perf/bgx_size/run_rung.sh "$card" "${foot:-t}$3_$1_$2" \
        --target "$1" --binder "$2" --rounds "$rounds" --trajectories 1 \
        ${foot:+--footprint} >> "$log" 2>&1
    echo "=== rc=$? ===" >> "$log"
}
run hPCSK9 100 704
run hTF    50  736
run hTF    100 800
run hTF    150 832
echo "=== TOP LADDER DONE $(date -u +%FT%TZ) ===" >> "$log"
