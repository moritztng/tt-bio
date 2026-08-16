#!/bin/bash
# Boltz-2 stage split on the Wormhole Galaxy, by ablation at fold level. Deliverable spine.
#
# base - R0 = the three recycles. base - S20 = 180 diffusion steps. R0S20 = the residue
# (embedder + one trunk pass + 20 steps + confidence + host). Nothing sits between the clock and
# the work: the two loop counts are tt_baseline module globals and xmodel_ab.py's --recycles and
# --steps set them after _resolve_*, so an unset flag reproduces the shipped value.
#
# baseAA is a second identical base arm: the A/A floor every delta here is judged against.
# Report the closure error base - (R0S20 + 3*per_recycle + 180*per_step); if it exceeds the floor
# the stages are not additive and the split is not usable, which is a result and must be said.
#
# Arms can straddle chips (wh_arm.sh picks per arm), so each records its card. Measured, not
# assumed: the 576 aa round put four control arms on three cards and they spanned 1.27 %, while
# the smallest delta this split resolves is tens of seconds.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}"
SIZE=${SIZE:-512}
REP=${REP:-2}
RETRIES=${RETRIES:-60}
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh
. "$TREE/perf/whb2/wh_arm.sh"

split_arm() {  # label recycles steps
  run_arm "$1" "$OUT" "$RETRIES" -- --model boltz2 --size "$SIZE" \
    --recycles "$2" --steps "$3" --repeat "$REP"
}

split_arm "split_base_$SIZE"   3 200
split_arm "split_R0_$SIZE"     0 200
split_arm "split_S20_$SIZE"    3  20
split_arm "split_R0S20_$SIZE"  0  20
split_arm "split_baseAA_$SIZE" 3 200
echo "SPLIT $SIZE DONE $(date -u +%H:%M:%S)"
