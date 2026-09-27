#!/bin/bash
# of3t-p10instr: the amended instrument (AMENDMENT-one-process-per-checkpoint.md). Each arm is
# scored ALONE in a fresh process, warmed on its own weights, twice on two different cards.
#   chainG.sh <card> <suffix> <arm> [<arm> ...]     arm = RUN/STEP, e.g. TF7/1
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
L=/home/ttuser/of3t_p10instr/logs
R=/home/ttuser/of3t_p10trainout/runs
C=/home/ttuser/of3t_p10trainout/corpus
cd "$W" || exit 1
card=$1; sfx=$2; shift 2
for arm in "$@"; do
    w=${arm%/*}; step=$(printf %08d "${arm#*/}")
    tag=G_${w}_$sfx
    echo "=== $tag start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) card $card" >> "$L/chainG.log"
    env TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:of3t-p10instr \
        timeout 1800 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_p10instr/evalnoise.py \
        --train-corpus $C/train12 --eval-corpus $C/val \
        --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt --seeds 20260926,0-31 \
        --adapter $R/$w/adapter-$step.safetensors \
        --out perf/of3t_p10instr/out/$tag.json > "$L/$tag.log" 2>&1
    echo "=== $tag done $(date -u +%FT%TZ) rc=$?" >> "$L/chainG.log"
done
echo "=== CHAING $sfx DONE $(date -u +%FT%TZ)" >> "$L/chainG.log"
