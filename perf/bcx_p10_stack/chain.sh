#!/bin/bash
# Interleaved A/B over the composed stack: all three levers off (= origin/main) against all
# three on, alternating on one card in one sitting so both arms see the same host load.
#   chain.sh <cycles> <rounds per arm> [arm spec...]
# An arm spec is `name:extra:template:triatt[:bw]`, where `bw` sends the triangle-attention
# backward through the fused kernel too. Default is the two-arm headline.
# TAG_PREFIX puts a chain's arms in their own subdirectory, so compare.py can be pointed at
# one chain instead of pooling every arm this row has ever run.
set -euo pipefail
cd "$(dirname "$0")/../.."
cycles=${1:-2}; rounds=${2:-6}; shift 2 || true
arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(off:0:0:0 on:1:1:1)
log=perf/bcx_p10_stack/out/chain.log
mkdir -p perf/bcx_p10_stack/out
for c in $(seq 1 "$cycles"); do
    for spec in "${arms[@]}"; do
        IFS=: read -r name e t r bw <<< "$spec"
        tag=${TAG_PREFIX:-}c${c}_${name}
        echo "=== $(date -u +%FT%TZ) $tag ($spec) ===" >> "$log"
        mkdir -p "$(dirname "perf/bcx_p10_stack/out/$tag")"
        bash perf/bcx_p10_stack/arm.sh "$tag" "$rounds" "$e" "$t" "$r" \
            --triatt-bw "${bw:-0}" \
            > "perf/bcx_p10_stack/out/$tag.log" 2>&1 || \
            echo "ARM $tag exited $?" >> "$log"
        grep -E '"(wall_seconds|stopped)"' "perf/bcx_p10_stack/out/$tag.log" >> "$log" || true
    done
done
echo "=== $(date -u +%FT%TZ) chain done ===" >> "$log"
