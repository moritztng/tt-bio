#!/bin/bash
# The rne_add round A/B: four processes in one sitting, arms alternating at the process
# boundary so a drift over the sitting cancels between them.
#
#   ab.sh [rounds] [tag-suffix]
#
# NINE rounds a process is the ceiling and it is not a sample-size choice: BindCraft 2's compile
# path deadlocks against itself at round 10 of this configuration against a cold cache, every
# time, on the same program hash (`state/perf10/bcx-p10-rne.md`). Round 1 is the compile and is
# dropped, so a process gives 8 timed rounds and the sitting gives 16 an arm.
#
# `off, on, on, off` and not `off, on, off, on`: the pairing that cancels a monotone drift is
# the symmetric one.
set -euo pipefail
cd "$(dirname "$0")/../.."
rounds=${1:-9}
suffix=${2:-$(date +%H%M%S)}
export ARM_OUT_ROOT=perf/bcx_p10_rneker/out
export ARM_XLA_CACHE=$PWD/perf/bcx_p10_rneker/out/xlacache
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-rneker
for arm in off on on off; do
    tag=p_${arm}_${suffix}_$RANDOM
    echo "=== $tag  rounds=$rounds  rne-kernel=$arm  $(date -u +%FT%TZ)" >&2
    bash perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi \
        --rne-kernel "$([ "$arm" = on ] && echo 1 || echo 0)" \
        > "perf/bcx_p10_rneker/out/$tag.log" 2>&1
done
