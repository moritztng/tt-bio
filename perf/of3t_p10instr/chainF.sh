#!/bin/bash
# of3t-p10instr: is ONE checkpoint per process, warmed on its own weights, repeatable?
# L40G and L40Gs1 (not A/B arms) each scored alone in two separate processes, one per card.
#   chainF.sh <card> <gate-line-in-chainAB.log> <suffix>
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
L=/home/ttuser/of3t_p10instr/logs
R=/home/ttuser/of3t_p10trainout/runs
C=/home/ttuser/of3t_p10trainout/corpus
cd "$W" || exit 1
card=$1; gate=$2; sfx=$3
until grep -q "$gate" "$L/chainAB.log" 2>/dev/null; do sleep 15; done
for w in L40G L40Gs1; do
    tag=F_${w}_$sfx
    echo "=== $tag start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) card $card" >> "$L/chainF.log"
    env TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:of3t-p10instr \
        timeout 1800 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_p10instr/evalnoise.py \
        --train-corpus $C/train12 --eval-corpus $C/val \
        --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt --seeds 20260926,0-31 \
        --adapter $R/$w/adapter-00000039.safetensors \
        --out perf/of3t_p10instr/out/$tag.json > "$L/$tag.log" 2>&1
    echo "=== $tag done $(date -u +%FT%TZ) rc=$?" >> "$L/chainF.log"
done
