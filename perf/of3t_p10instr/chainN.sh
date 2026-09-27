#!/bin/bash
# of3t-p10instr noise characterisation, qb2 card 2. Weights that are NOT an A/B arm only.
#   N3a: base, L40G-39, L40Gs1-39 over eval seeds 20260926 + 0..31
#   N3b: the same three in reverse order, a second process: how far process history moves it
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
L=/home/ttuser/of3t_p10instr/logs
R=/home/ttuser/of3t_p10trainout/runs
C=/home/ttuser/of3t_p10trainout/corpus
cd "$W" || exit 1
run () {
    tag=$1; shift
    echo "=== $tag start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" >> "$L/chainN.log"
    env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:of3t-p10instr \
        timeout 3000 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_p10instr/evalnoise.py \
        --train-corpus $C/train12 --eval-corpus $C/val \
        --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
        --seeds 20260926,0-31 --out perf/of3t_p10instr/out/$tag.json "$@" > "$L/$tag.log" 2>&1
    echo "=== $tag done $(date -u +%FT%TZ) rc=$?" >> "$L/chainN.log"
}
run N3a --adapter base --adapter $R/L40G/adapter-00000039.safetensors --adapter $R/L40Gs1/adapter-00000039.safetensors
run N3b --adapter $R/L40Gs1/adapter-00000039.safetensors --adapter $R/L40G/adapter-00000039.safetensors --adapter base
