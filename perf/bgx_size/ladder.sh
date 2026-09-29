#!/bin/bash
# The timed leg of the size ladder, smallest rung first, one card, one trajectory each.
#
# Sequential and smallest-first on purpose: the large rungs are expected to refuse, and a
# rung that dies must not take the rungs below it with it. Every rung writes its own
# rung.json before the next one starts.
#
# One trajectory, so the round time is one trajectory's round and comparable with
# bcx-bigtarget's curve. What the shipped `auto` default would have chosen is recorded in
# every rung.json as `auto_would_choose` whatever this leg passes.
set -u
cd "$(dirname "$0")/../.."
card=${1:-1}
rounds=${2:-4}
log=perf/bgx_size/out/ladder_card$card.log
mkdir -p perf/bgx_size/out
: > "$log"
run() {   # run <target> <binder> <tokens>
    echo "=== $(date -u +%FT%TZ) rung $1/$2 -> $3 tokens ===" >> "$log"
    timeout 3600 perf/bgx_size/run_rung.sh "$card" "t$3_$1_$2" \
        --target "$1" --binder "$2" --rounds "$rounds" --trajectories 1 >> "$log" 2>&1
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
echo "=== LADDER DONE $(date -u +%FT%TZ) ===" >> "$log"
