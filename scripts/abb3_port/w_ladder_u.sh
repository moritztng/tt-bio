#!/bin/bash
# The UPRUNG arm: the same pinned ladder on train-u-relpos-ondevice merged into the same base,
# so the only difference between the two runs is U's on-card feature expansion. Thread width is
# pinned the same way (16 host cores divided by the arm), because an unpinned world-4 arm spends
# 914 s of a 931 s step in host torch and no lever is scoreable against that.
# Pre-registered prediction: if host contention is the 4-chip ceiling, U helps MORE at 4 than at 2.
cd /home/ttuser/w-uprung
source /home/ttuser/tt-bio-dev/env/bin/activate
export PYTHONPATH=$PWD
export TT_BIO_LEASE_CARDS=0,1,2,3
export TT_BIO_LEASE_HOLDER=worker:train-w-fourchip
rm -rf /dev/shm/abb3-w-uprung runs/w-ladder-u
exec python3 -u scripts/abb3_port/dp_gate.py \
    --out runs/w-ladder-u --chips 1,2,3,0 --ladder 1,2,4 --torch-threads 16 \
    --steps 4 --rounds 3 --micro 4 --global-batch 64 --tokens 256 --blocks 8 --seed 0 \
    --data sabdab --split train --rendezvous /dev/shm/abb3-w-uprung \
    --json perf/train_w_fourchip/ladder_uprung.json
