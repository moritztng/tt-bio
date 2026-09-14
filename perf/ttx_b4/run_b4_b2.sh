#!/bin/bash
# Catalogue row B4 at the fold. `kb2` is the lever the row names (halve every `K_block`);
# `mmretune` is the bit-exact control the production-shape screen produced instead. Arms are
# interleaved on,kb2,mmretune,on,kb2,mmretune so each pair carries its own A/A.
WT=/home/ttuser/.coworker/wt/ttx-b4-genericop-kblock-ship
cd $WT
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:ttx-b4-genericop-kblock-ship PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
PY=/home/ttuser/tt-bio-dev/env/bin/python3
MODEL=${1:-boltz2}
/home/ttuser/.coworker/scripts/benchlock.sh ttx-b4-genericop-kblock-ship -- \
  $PY -u perf/other512/fold_ab_multi.py --model $MODEL --sizes 512 \
      --arms on,kb2,mmretune,on,kb2,mmretune --out perf/ttx_b4/ab_${MODEL}_512_qb2c3.json
echo "RC=$?"
