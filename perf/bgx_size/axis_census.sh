#!/bin/bash
# The true Evoformer token axis for every (target, binder) pair in the ladder.
#
# `tokens` in every rung so far is _pad32(target_residues + binder) off targets.json. That is
# exact for a single-chain target and WRONG for a multi-chain one: the fused-arm note printed
# 544 on a rung labelled 512 (hIL2R, 2 chains) and 608 on one labelled 576 (hTNFa, 3 chains),
# both one bucket high, while hPDL1 on one chain matched. BindCraft 2 hands the splice whatever
# complex it built and `_pad` buckets THAT. One round is enough: the axis is fixed at the first
# forward, so this is the cheapest way to relabel every rung in the table correctly.
#
#   axis_census.sh <card> <tag>
set -u
cd "$(dirname "$0")/../.."
card=$1; tag=$2
log=perf/bgx_size/out/$tag.log
mkdir -p perf/bgx_size/out
: > "$log"
for spec in "hPDL1 50" "hPDL1 141" "hPDL1 146" "hIL7RA 100" "hCA2 100" "hCA2 150" \
            "hIL2R 50" "hIL2R 90" "hIL2R 100" "hIL2R 146" "hTNFa 50" "hTNFa 100" "hHSA 100"; do
    set -- $spec
    echo "=== $(date -u +%FT%TZ) axis $1/$2 ===" >> "$log"
    timeout 1800 perf/bgx_size/run_rung.sh "$card" "ax_$1_$2" \
        --target "$1" --binder "$2" --rounds 1 --trajectories 1 >> "$log" 2>&1
    echo "=== rc=$? ===" >> "$log"
done
echo "=== $tag DONE $(date -u +%FT%TZ) ===" >> "$log"
