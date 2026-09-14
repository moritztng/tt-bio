#!/bin/bash
# Detached: a foreground ssh-launched leg dies on a broken pipe (SIGHUP killed r2 after 6 of 15).
WT=/home/ttuser/.coworker/wt/ttx-b4-genericop-kblock-ship
cd $WT
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:ttx-b4-genericop-kblock-ship
# Refuse to measure while a sibling fold is co-resident. benchlock's default is to WARN and
# proceed after 900 s, and that default is wrong for this question: the r2 run took its quiet clear
# in the GAP BETWEEN two of ttx-a3's 1536 aa legs, a3 started its next leg, and the fold came out
# 23.5 s against 15.8 s on the same fixture in an uncontended run. A 1.49x absolute shift swamps
# the 1-4 % being measured, so 5400 s here means this run waits for a genuinely quiet box or
# produces nothing at all. Nothing is the better of the two outcomes.
export BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=5400
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
/home/ttuser/.coworker/scripts/benchlock.sh ttx-b4-genericop-kblock-ship -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py --model boltz2 \
    --sizes 512 --arms on,kb2,mmretune,on,kb2,mmretune,on,kb2,mmretune \
    --out perf/ttx_b4/ab_boltz2_512_qb2c3_r3.json
echo "RC=$?"
