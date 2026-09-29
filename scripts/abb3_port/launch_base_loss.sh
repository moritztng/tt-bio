#!/bin/bash
cd /home/ttuser/.coworker/wt/train-i-run
source /home/ttuser/tt-bio-dev/env/bin/activate
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=2,3 TT_BIO_LEASE_CARDS=2,3 TT_BIO_LEASE_HOLDER=worker:train-i-run
exec python3 scripts/abb3_port/supervise.py \
    --out runs/base-loss --steps 193512 --chips 2,3 \
    --micro 4 --global-batch 64 --tokens 256 --blocks 8 --seed 0 \
    --data sabdab --split train --days 5 \
    --lease-holder worker:train-i-run --rendezvous /dev/shm/abb3-base-loss "$@"
