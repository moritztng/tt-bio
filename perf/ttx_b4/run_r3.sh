#!/bin/bash
# Detached: a foreground ssh-launched leg dies on a broken pipe (SIGHUP killed r2 after 6 of 15).
WT=/home/ttuser/.coworker/wt/ttx-b4-genericop-kblock-ship
cd $WT
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:ttx-b4-genericop-kblock-ship
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
/home/ttuser/.coworker/scripts/benchlock.sh ttx-b4-genericop-kblock-ship -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py --model boltz2 \
    --sizes 512 --arms on,kb2,mmretune,on,kb2,mmretune,on,kb2,mmretune \
    --out perf/ttx_b4/ab_boltz2_512_qb2c3_r3.json
echo "RC=$?"
