#!/bin/bash
# Interleaved A/B: the extra-MSA stack in JAX on the host (A) against the same stack on card
# (B), alternating on one card in one sitting so the host load both arms see is the same load.
#   chain.sh <cycles> <rounds per arm>
# Two cycles of 6 rounds gives 5 timed rounds per arm per cycle: `analyze.py` drops the round
# that carries BindCraft 2's jit compile, and a round needs a closing boundary to have a wall.
set -euo pipefail
cd "$(dirname "$0")/../.."
cycles=${1:-2}; rounds=${2:-6}
log=perf/bcx_p10_resident/out/chain.log
mkdir -p perf/bcx_p10_resident/out
: > "$log"
for c in $(seq 1 "$cycles"); do
    for arm in 0 1; do
        tag=ab_c${c}_x${arm}
        echo "=== $(date -u +%FT%TZ) $tag ===" >> "$log"
        bash perf/bcx_p10_resident/arm.sh "$tag" "$rounds" "$arm" \
            > "perf/bcx_p10_resident/out/$tag.log" 2>&1 || \
            echo "ARM $tag exited $?" >> "$log"
        grep -E '"(wall_seconds|stopped)"' "perf/bcx_p10_resident/out/$tag.log" >> "$log" || true
    done
done
echo "=== $(date -u +%FT%TZ) chain done ===" >> "$log"
