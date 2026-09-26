#!/bin/bash
# Both accuracy legs of the wave-10 stack, on a qb2 card.
#   acc.sh <card> [rounds] [struct|vjp|struct,vjp]
#
# NOT on pc card 0, and this is not a preference. That p150a silently computes some ttnn
# MATMULS wrong -- linear_z differs on 15/15 repeats at 256 aa fp32 and 2/31 at bf16, the
# transfer path is bit-exact and concat/layer_norm are stable, and it is location-keyed rather
# than size-keyed (memory `pc-card0-512aa-fold-nondeterminism`, root-caused 2026-08-17). The
# rule is that card must not host bit-exact or hash-equality gating at any size. The TIMING legs
# stay there: a miscomputed matmul takes the same time as a correct one.
#
# Leg 1, structure in Angstrom. Three arms, one card, serially: the anchor twice for this
# fixture's own A/A floor, then the composed stack. Round 1 is the only paired reading -- from
# round 2 the optimiser has acted on a different gradient and the arms hold different sequences.
# Leg 2, the float64 VJP at the composed configuration, both arms.
set -euo pipefail
cd "$(dirname "$0")/../.."
card=${1:?usage: acc.sh <card> [rounds] [legs]}
rounds=${2:-4}
legs=${3:-struct,vjp}        # either leg alone, so one can be retaken without the other
export ARM_OUT_ROOT=$PWD/perf/bcx_p10_stack2/out_acc
export ARM_XLA_CACHE=$PWD/perf/bcx_p10_stack2/out_acc/xlacache
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack2
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
mkdir -p "$ARM_OUT_ROOT"
# `save_design_frames` is what makes BindCraft 2 write a CIF per round
# (`bindcraft/trajectory.py:283` -> `TrajectoryRecorder.keep_frames`). Without it a 4-round arm
# that never reaches the end of its stage writes no structure at all and the leg scores nothing.
FRAMES="--set save_design_frames=1"
[ "${legs/struct/}" = "$legs" ] || for a in acc_off_a:0 acc_off_b:0 acc_on:1; do
    tag=${a%%:*}; on=${a##*:}
    echo "=== $(date -u +%FT%TZ) $tag (levers=$on) ==="
    if [ "$on" = 1 ]; then
        TT_BIO_MM_LAYOUT=1 perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi \
            --triatt-bw 1 --rne-kernel 1 $FRAMES
    else
        TT_BIO_MM_LAYOUT=0 perf/bcx_p10_stack/arm.sh "$tag" "$rounds" 1 1 hifi $FRAMES
    fi
done
[ "${legs/vjp/}" = "$legs" ] || for arm in off on; do
    echo "=== $(date -u +%FT%TZ) vjp $arm ==="
    out=$ARM_OUT_ROOT/grad_$arm; mkdir -p "$out"
    PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card \
        /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_stack2/grad_stack2.py \
        --arm "$arm" --out "$out"
done
echo "=== $(date -u +%FT%TZ) accuracy legs complete ==="
