#!/bin/bash
# Interleaved A/B: the multimer template embedding as the model computes it (0) against the same
# round with it forced to a constant (1), alternating on one card in one sitting so both arms
# see the same host load. The round is 63 % host and load-sensitive, so two runs an hour apart
# would measure the box.
#   chain.sh <cycles> <rounds per arm>
set -euo pipefail
cd "$(dirname "$0")/../.."
cycles=${1:-2}; rounds=${2:-6}
log=perf/bcx_p10_tmplemb/out/chain.log
mkdir -p perf/bcx_p10_tmplemb/out
: > "$log"
for c in $(seq 1 "$cycles"); do
    for arm in 0 1; do
        tag=ab_c${c}_t${arm}
        echo "=== $(date -u +%FT%TZ) $tag ===" >> "$log"
        bash perf/bcx_p10_tmplemb/arm.sh "$tag" "$rounds" "$arm" \
            > "perf/bcx_p10_tmplemb/out/$tag.log" 2>&1 || \
            echo "ARM $tag exited $?" >> "$log"
        grep -E '"(wall_seconds|stopped)"' "perf/bcx_p10_tmplemb/out/$tag.log" >> "$log" || true
    done
done
echo "=== $(date -u +%FT%TZ) chain done ===" >> "$log"
