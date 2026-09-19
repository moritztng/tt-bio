#!/bin/bash
# Census arm, deliberately OUTSIDE benchlock. A call count does not move with host load, and
# the campaign ledger admits a census under contention where it refuses a timed A/B. Taking the
# lock for this would block pvx-baseline, whose clocked cell genuinely needs it. The fold
# SECONDS this run records are therefore not admissible as a perf number and are labelled so.
set -u
WT=/home/ttuser/.coworker/wt/pvx-inventory
cd "$WT" || exit 1
MODEL="$1"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pvx-inventory
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
/home/ttuser/tt-bio-dev/env/bin/python3 -u perf/pvx_inventory/firing.py \
  --model "$MODEL" --size 512 --card 0 \
  --out "perf/pvx_inventory/firing_${MODEL}_512_qb1c0.json"
echo "RC=$?"
