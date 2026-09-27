#!/bin/bash
# The milestone accuracy legs on the wave-15 stack, one qb2 card, serially.
#   acc.sh [rounds] [struct|vjp|struct,vjp]
#
# Structure: `perf/bcx_p10_stack2/acc.sh`'s layout. The hifi anchor twice (this fixture's A/A
# floor), then the full stack; round 1 is the only paired reading. VJP: `grad_stack2.py --arm on`
# with channel move and widen_add armed from the environment, and the `off` anchor beside it.
set -euo pipefail
cd "$(dirname "$0")/../.."
rounds=${1:-4}; legs=${2:-struct,vjp}
export ARM_OUT_ROOT=$PWD/perf/bcx_p10_stack5/out_acc
export ARM_XLA_CACHE=$PWD/perf/bcx_p10_stack5/out_acc/xlacache
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack5
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
mkdir -p "$ARM_OUT_ROOT"
FRAMES="--set save_design_frames=1"
STACK="TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1 TT_BIO_GRAD_FANIN_L1=0"
ANCHOR="TT_BIO_MM_LAYOUT=0 TT_BIO_TAPED_CHANNEL_MOVE=0 TT_BIO_WIDEN_ADD=0 TT_BIO_GRAD_FANIN_L1=0"
[ "${legs/struct/}" = "$legs" ] || for a in acc_off_a:0 acc_off_b:0 acc_on:1; do
    tag=${a%%:*}; on=${a##*:}
    echo "=== $(date -u +%FT%TZ) $tag (levers=$on) ==="
    if [ "$on" = 1 ]; then
        env $STACK perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi \
            --triatt-bw 1 --rne-kernel 1 $FRAMES
    else
        env $ANCHOR perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi $FRAMES
    fi
done
[ "${legs/vjp/}" = "$legs" ] || for arm in off on; do
    echo "=== $(date -u +%FT%TZ) vjp $arm ==="
    out=$ARM_OUT_ROOT/grad_$arm; mkdir -p "$out"
    if [ "$arm" = on ]; then e=$STACK; else e=$ANCHOR; fi
    env $e PYTHONPATH=$PWD /home/ttuser/bcx_e2e_venv/bin/python3 -u \
        perf/bcx_p10_stack2/grad_stack2.py --arm "$arm" --out "$out"
done
echo "=== $(date -u +%FT%TZ) accuracy legs complete ==="
