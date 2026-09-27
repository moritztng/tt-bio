#!/bin/bash
# of3t-p10instr: the amended instrument (AMENDMENT-one-process-per-checkpoint.md). Each arm is
# scored ALONE in a fresh process, warmed on its own weights, twice on two different cards.
#   chainG.sh [-n DRAWS] <card> <suffix> <arm> [<arm> ...]     arm = RUN/STEP, e.g. TF7/1
# -n scores eval seeds 0..DRAWS-1 (default 32) into D<DRAWS>_<arm>_<suffix>.json; the call sequence
# up to seed 31 is the 32-draw one, so those draws must repeat the G_ artifacts digit for digit.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
L=/home/ttuser/of3t_p10instr/logs
R=/home/ttuser/of3t_p10trainout/runs
C=/home/ttuser/of3t_p10trainout/corpus
cd "$W" || exit 1
draws=32; [ "${1:-}" = -n ] && { draws=$2; shift 2; }
pre=G; [ "$draws" = 32 ] || pre=D$draws
holder=${TT_BIO_LEASE_HOLDER:-worker:of3t-p10instr}
card=$1; sfx=$2; shift 2
for arm in "$@"; do
    w=${arm%/*}; step=$(printf %08d "${arm#*/}")
    tag=${pre}_${w}_$sfx
    echo "=== $tag start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) card $card" >> "$L/chainG.log"
    env TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=$holder \
        timeout 1800 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_p10instr/evalnoise.py \
        --train-corpus $C/train12 --eval-corpus $C/val \
        --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt --seeds 20260926,0-$((draws - 1)) \
        --adapter $R/$w/adapter-$step.safetensors \
        --out perf/of3t_p10instr/out/$tag.json > "$L/$tag.log" 2>&1
    rc=$?
    echo "=== $tag done $(date -u +%FT%TZ) rc=$rc" >> "$L/chainG.log"
done
echo "=== CHAING $pre $sfx DONE $(date -u +%FT%TZ)" >> "$L/chainG.log"
