#!/bin/bash
# One model, one 512 aa fold pair (cold + warm), counters read off the warm fold.
set -u
WT=/home/ttuser/.coworker/wt/pvx-inventory
cd "$WT" || exit 1
MODEL="$1"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pvx-inventory
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export BENCHLOCK_MAXLOAD=1.5
/home/ttuser/.coworker/scripts/benchlock.sh pvx-inventory -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/pvx_inventory/firing.py \
    --model "$MODEL" --size 512 --card 0 \
    --out "perf/pvx_inventory/firing_${MODEL}_512_qb1c0.json"
echo "RC=$?"
