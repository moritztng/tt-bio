#!/bin/bash
# One sitting, three arms, six processes.
#   sit.sh [tag-prefix] [order...]      default: p  off cmp on on cmp off
#
# The order is a palindrome, so every arm's mean process position is 2.5 and a monotone drift
# over the sitting cancels for all three equally rather than loading one. Arms alternate at the
# PROCESS boundary, never inside one, because the levers are resolved at import.
#
# The second sitting runs the palindrome REVERSED (`sit.sh q on cmp off off cmp on`). Pooled,
# every arm then occupies each end of a sitting exactly once, so anything that depends on where
# an arm sits -- a warming cache, a co-tenant that arrives -- cancels across the two.
#
# 9 rounds a process, round 1 of each dropped as its compile round: 8 timed rounds a process,
# 16 an arm. Nine is the ceiling, see arm.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
r=${ROUNDS:-9}
prefix=${1:-p}; shift || true
order=("${@:-}"); [ $# -eq 0 ] && order=(off cmp on on cmp off)
i=0
for mode in "${order[@]}"; do
    i=$((i + 1)); tag=${prefix}${i}_${mode}
    echo "=== $(date -u +%H:%M:%SZ) $tag ($mode) load1=$(cut -d' ' -f1 /proc/loadavg) ==="
    perf/bcx_p10_stack3/arm.sh "$tag" "$r" "$mode"
done
echo "=== $(date -u +%H:%M:%SZ) sitting complete ==="
