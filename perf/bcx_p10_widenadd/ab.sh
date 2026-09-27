#!/bin/bash
# bcx-p10-widenadd: the round A/B for TT_BIO_WIDEN_ADD on the composed stack.
#
#   ab.sh [rounds] [tag-suffix] [arms...]
#
# `perf/bcx_p10_l1fuse/ab.sh` with this row's card, holder, output dir and base. Base arm =
# stack3 (`--triatt-bw 1 --rne-kernel 1`, hifi route, extra-MSA and template on,
# TT_BIO_MM_LAYOUT=1) + TT_BIO_TAPED_CHANNEL_MOVE=1. TT_BIO_GRAD_FANIN_L1 stays off on both arms:
# `bcx-p10-stack4` had not read GO when this sitting was set up, and on every call the kernel
# serves there is no widened tensor left for it to place.
#
# Nine rounds a process (BindCraft 2's compile path deadlocks at round 10 on a cold cache),
# round 1 of each dropped, arms alternating at the process boundary.
set -euo pipefail
cd "$(dirname "$0")/../.."
rounds=${1:-9}
suffix=${2:-$(date +%H%M%S)}
shift 2 || true
arms=("$@")
[ $# -eq 0 ] && arms=(off on on off)
out=perf/bcx_p10_widenadd/out
export ARM_OUT_ROOT=$out
export ARM_XLA_CACHE=$PWD/$out/xlacache
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-widenadd
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-1} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-1}
mkdir -p "$out"
for arm in "${arms[@]}"; do
    tag=p_${arm}_${suffix}_$RANDOM
    echo "=== $tag rounds=$rounds TT_BIO_WIDEN_ADD=$arm $(date -u +%FT%TZ)" >&2
    env TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_GRAD_FANIN_L1=0 \
        TT_BIO_WIDEN_ADD=$([ "$arm" = on ] && echo 1 || echo 0) \
        bash perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi \
            --triatt-bw 1 --rne-kernel 1 \
        > "$out/$tag.log" 2>&1
    echo "=== $tag done $(date -u +%FT%TZ)" >&2
done
echo "=== all arms done $(date -u +%FT%TZ)" >&2
