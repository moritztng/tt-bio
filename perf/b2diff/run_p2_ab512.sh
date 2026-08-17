#!/bin/bash
# The page cell, remeasured. The published 24.822 s Boltz-2 512 aa cell was measured on THIS card
# (qb2 card 0, ttnn 0.68.0, benchlock, cdk2x2_512, warm, median of n=3 -- state/boltz2-s2-extract-land.md),
# so `on` here reproduces the published baseline and `l7l6` is the number the new defaults ship.
# Arms alternated, one process, no region timers: the page number is a bare fold wall.
WT=/home/ttuser/.coworker/wt/boltz2-diffusion-perf-p2-land
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf-p2-land PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=900
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf-p2-land -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/b2diff/fold_ab_b2diff.py \
    --sizes 512 --arms on,l7l6,on,l7l6,on,l7l6 --no-timers \
    --out perf/b2diff/p2_ab512.json
echo "RC=$?"
