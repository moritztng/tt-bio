#!/bin/bash
# Pass-2 parity preflight on qb2 card 0 (grid 11x10, ttnn 0.68.0). The predecessor proved L7+L6
# bit-exact on qb1 card 1 (13x10); this re-proves it on the card the page cell is measured on and
# on the grid the shipped default will run.
WT=/home/ttuser/.coworker/wt/boltz2-diffusion-perf-p2-land
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf-p2-land PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=3600 BENCHLOCK_LOAD_WAIT_S=300
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf-p2-land -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/b2diff/fold_ab_b2diff.py \
    --sizes 298 --arms on,l7l6,on,l7l6 --no-timers \
    --out perf/b2diff/p2_ab298.json
echo "RC=$?"
