#!/bin/bash
# The controlled re-take: each rank gets the host's cores divided by the arm's width, instead
# of torch's default, which is the whole box per rank. Needs a free qb1 -- chip 0 is node 1 and
# trix-writer-split-upstream-pr takes it for matmul suites.
cd /home/ttuser/.coworker/wt/train-w-fourchip
source /home/ttuser/tt-bio-dev/env/bin/activate
export PYTHONPATH=$PWD
export TT_BIO_LEASE_CARDS=0,1,2,3
export TT_BIO_LEASE_HOLDER=worker:train-w-fourchip
rm -rf /dev/shm/abb3-w-pinned runs/w-ladder-pinned
exec python3 -u scripts/abb3_port/dp_gate.py \
    --out runs/w-ladder-pinned --chips 1,2,3,0 --ladder 1,2,4 --torch-threads 16 \
    --steps 4 --rounds 2 --micro 4 --global-batch 64 --tokens 256 --blocks 8 --seed 0 \
    --data sabdab --split train --rendezvous /dev/shm/abb3-w-pinned \
    --json perf/train_w_fourchip/ladder_pinned.json
