#!/bin/bash
# Interleaved A/B: the multimer template pair stack on card (1) against the same round with it
# in BindCraft 2's JAX (0), alternating on one card in one sitting.
#   devchain.sh <cycle label> <rounds per arm> [arm order, default "1 0"]
# The order is an argument because one cycle cannot tell a lever from a drift in the box's
# load, and this round is 63 % host: run cycle 2 as "0 1" so a monotonic drift lands on the
# other arm. Each arm is wrapped in `timeout` so a dropped launcher leaves no orphan holding
# the card.
set -euo pipefail
cd "$(dirname "$0")/../.."
cycle=${1:-1}; rounds=${2:-6}; order=${3:-1 0}
log=perf/bcx_p10_tmplemb/out/devchain.log
mkdir -p perf/bcx_p10_tmplemb/out
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-tmplemb
for arm in $order; do
    tag=dev_c${cycle}_d${arm}
    out=perf/bcx_p10_tmplemb/out/$tag
    rm -rf "$out"; mkdir -p "$out"
    echo "=== $(date -u +%FT%TZ) $tag ===" >> "$log"
    timeout 900 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_tmplemb/devarm.py \
        --template "$arm" --rounds "$rounds" --exact 0 --shipped --binder 146 \
        --out "$out" > "perf/bcx_p10_tmplemb/out/$tag.log" 2>&1 \
        || echo "ARM $tag exited $?" >> "$log"
    grep -o '"template_device": {.*}' "perf/bcx_p10_tmplemb/out/$tag.log" | tail -1 >> "$log"
done
echo "=== $(date -u +%FT%TZ) cycle $cycle done ===" >> "$log"
