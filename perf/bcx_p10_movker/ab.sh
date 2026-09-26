#!/bin/bash
# Leg 5: the pair-transpose dispatch gate, A/B'd at the ROUND on the composed tree.
#
# Four processes alternating off, on, on, off at the PROCESS boundary, 9 requested rounds each
# (the BindCraft 2 compile deadlocks against itself at round 10 of this configuration), round 1
# dropped as that process's compile, 8 timed. The palindrome is what cancels a drift in the box
# across the run; two processes would hit the 8-rounds-an-arm bar and cancel nothing.
#
# Both arms are the composed round -- triatt-bw, rne-kernel, TT_BIO_MM_LAYOUT and the hifi route
# all ON -- because a lever measured against the pre-wave-10 anchor is measuring wave 10 again.
# `--pt-rm-min-c` is the ONLY difference between the arms.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${ROUNDS:-9}
CARD=${CARD:-1}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-movker
export TT_BIO_MM_LAYOUT=1
export ARM_OUT_ROOT=perf/bcx_p10_movker/out
export ARM_XLA_CACHE=$PWD/perf/bcx_p10_movker/out/xlacache
mkdir -p perf/bcx_p10_movker/out

for spec in off:0 on:256 on:256 off:0; do
    arm=${spec%%:*}; minc=${spec##*:}
    tag="ab_${arm}_$$_$(date +%H%M%S)"
    echo "=== $(date -u +%FT%TZ) arm=$arm min_c=$minc tag=$tag ===" >&2
    bash perf/bcx_p10_stack/arm.sh "$tag" "$ROUNDS" 1 1 hifi \
        --triatt-bw 1 --rne-kernel 1 --pt-rm-min-c "$minc" \
        >"perf/bcx_p10_movker/out/$tag.log" 2>&1 || echo "arm $tag exited $?" >&2
    echo "=== $(date -u +%FT%TZ) done $tag ===" >&2
done
echo "ALL ARMS DONE $(date -u +%FT%TZ)" >&2
