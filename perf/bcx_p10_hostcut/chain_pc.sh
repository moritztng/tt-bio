#!/bin/bash
# Interleaved arms on pc, alternating at the process boundary so both routes see the same host
# load in one sitting. `perf/bcx_p10_stack/chain.sh` pointed at this row's arm and output dir.
#   chain_pc.sh <cycles> <rounds per arm> [name:extra:template:triatt ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
cycles=${1:-2}; rounds=${2:-10}; shift 2 || true
arms=("$@"); [ ${#arms[@]} -gt 0 ] || arms=(hifi:1:1:hifi agtri:1:1:agtri)
log=perf/bcx_p10_hostcut/out/chain_pc.log
for c in $(seq 1 "$cycles"); do
    for spec in "${arms[@]}"; do
        IFS=: read -r name e t r <<< "$spec"
        tag=c${c}_${name}
        echo "=== $(date -u +%FT%TZ) $tag ($spec) loadavg1=$(cut -d' ' -f1 /proc/loadavg) ===" >> "$log"
        bash perf/bcx_p10_hostcut/arm_pc.sh "$tag" "$rounds" "$e" "$t" "$r" \
            > "perf/bcx_p10_hostcut/out/$tag.log" 2>&1 || echo "ARM $tag exited $?" >> "$log"
        grep -E '"(wall_seconds|stopped)"' "perf/bcx_p10_hostcut/out/$tag.log" >> "$log" || true
    done
done
echo "=== $(date -u +%FT%TZ) chain done ===" >> "$log"
