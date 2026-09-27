#!/bin/bash
# bcx-p10-l1fuse leg 5: the round A/B for the fan-in L1 placement.
#
#   ab.sh [rounds] [tag-suffix] [arms...]
#
# The composed round on both arms -- `--triatt-bw 1 --rne-kernel 1`, the hifi route, extra-MSA
# and template on, TT_BIO_MM_LAYOUT=1 -- with TT_BIO_GRAD_FANIN_L1 the only difference. Holding
# mm_layout on both sides matters: it is a live module read as well as an env var, and an arm
# that armed it on one side only would measure two levers.
#
# NINE rounds a process is the ceiling and it is not a sample-size choice: BindCraft 2's compile
# path deadlocks against itself at round 10 of this configuration against a cold cache, every
# time, on the same program hash. Round 1 is the compile and is dropped, so a process gives 8
# timed rounds and four processes give 16.
#
# The arms alternate at the process boundary, off on on off, so a monotone drift over the
# sitting cancels rather than landing on one arm.
set -euo pipefail
cd "$(dirname "$0")/../.."
rounds=${1:-9}
suffix=${2:-$(date +%H%M%S)}
shift 2 || true
arms=("${@:-off on on off}")
[ $# -eq 0 ] && arms=(off on on off)
export ARM_OUT_ROOT=perf/bcx_p10_l1fuse/out
export ARM_XLA_CACHE=$PWD/perf/bcx_p10_l1fuse/out/xlacache
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcx-p10-l1fuse}
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
# Which env var the two arms toggle. Default is this rows own lever; the campaign also wants a
# device-column reading on TT_BIO_TAPED_CHANNEL_MOVE, which was declined on a wall number, and
# that is the same four processes with one name changed. A space-separated list arms every
# name in it together on the on arm and none of them on the off arm, which is how a stack of
# levers is priced as a stack.
LEVER=${LEVER:-TT_BIO_GRAD_FANIN_L1}
mkdir -p perf/bcx_p10_l1fuse/out
for arm in "${arms[@]}"; do
    tag=p_${arm}_${suffix}_$RANDOM
    echo "=== $tag rounds=$rounds [$LEVER]=$arm $(date -u +%FT%TZ)" >&2
    envs=(); for v in $LEVER; do envs+=("$v=$([ "$arm" = on ] && echo 1 || echo 0)"); done
    env TT_BIO_MM_LAYOUT=1 "${envs[@]}" \
        bash perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi \
            --triatt-bw 1 --rne-kernel 1 \
        > "perf/bcx_p10_l1fuse/out/$tag.log" 2>&1
    echo "=== $tag done $(date -u +%FT%TZ)" >&2
done
echo "=== all arms done $(date -u +%FT%TZ)" >&2
