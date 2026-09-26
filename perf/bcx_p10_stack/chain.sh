#!/bin/bash
# Interleaved A/B over the composed stack: all three levers off (= origin/main) against all
# three on, alternating on one card in one sitting so both arms see the same host load.
#   chain.sh <cycles> <rounds per arm> [arm spec...]
# An arm spec is `name:extra:template:triatt`. Default is the two-arm headline.
set -euo pipefail
cd "$(dirname "$0")/../.."
cycles=${1:-2}; rounds=${2:-6}; shift 2 || true
arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(off:0:0:0 on:1:1:1)
log=perf/bcx_p10_stack/out/chain.log
mkdir -p perf/bcx_p10_stack/out
for c in $(seq 1 "$cycles"); do
    for spec in "${arms[@]}"; do
        IFS=: read -r name e t r <<< "$spec"
        tag=c${c}_${name}
        echo "=== $(date -u +%FT%TZ) $tag ($spec) ===" >> "$log"
        bash perf/bcx_p10_stack/arm.sh "$tag" "$rounds" "$e" "$t" "$r" \
            > "perf/bcx_p10_stack/out/$tag.log" 2>&1 || \
            echo "ARM $tag exited $?" >> "$log"
        grep -E '"(wall_seconds|stopped)"' "perf/bcx_p10_stack/out/$tag.log" >> "$log" || true
    done
done
echo "=== $(date -u +%FT%TZ) chain done ===" >> "$log"
