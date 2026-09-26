#!/bin/bash
# Interleaved A/B: the multimer template pair stack on card (1) against the same round with it
# in BindCraft 2's JAX (0), alternating on one card in one sitting. The device arm goes first
# because it carries the longer first-round compile.
#   devchain.sh <cycles> <rounds per arm>
set -euo pipefail
cd "$(dirname "$0")/../.."
cycles=${1:-1}; rounds=${2:-6}
log=perf/bcx_p10_tmplemb/out/devchain.log
mkdir -p perf/bcx_p10_tmplemb/out
: > "$log"
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-tmplemb
for c in $(seq 1 "$cycles"); do
    for arm in 1 0; do
        tag=dev_c${c}_d${arm}
        out=perf/bcx_p10_tmplemb/out/$tag
        rm -rf "$out"; mkdir -p "$out"
        echo "=== $(date -u +%FT%TZ) $tag ===" >> "$log"
        /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_tmplemb/devarm.py \
            --template "$arm" --rounds "$rounds" --exact 0 --shipped --binder 146 \
            --out "$out" > "perf/bcx_p10_tmplemb/out/$tag.log" 2>&1 \
            || echo "ARM $tag exited $?" >> "$log"
        grep -o '"template_device": {.*}' "perf/bcx_p10_tmplemb/out/$tag.log" | tail -1 >> "$log"
    done
done
echo "=== $(date -u +%FT%TZ) devchain done ===" >> "$log"
