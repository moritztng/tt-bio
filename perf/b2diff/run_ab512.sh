#!/bin/bash
# The pre-registered fold-level A/B, one process, arms alternated so a monotonic drift cannot
# masquerade as a lever. Three `on` folds give the A/A floor; `l7l6` and `l7l6s6` are the
# integrated arms and are MEASURED, never quoted as a sum of the single-lever numbers.
WT=/home/ttuser/.coworker/wt/boltz2-diffusion-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=1800
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/b2diff/fold_ab_b2diff.py \
    --sizes 512 --arms on,l7,on,l6,l7l6,on,s6,l7l6s6,l6,l7,on,l7l6s6 \
    --out perf/b2diff/ab512.json
echo "RC=$?"
