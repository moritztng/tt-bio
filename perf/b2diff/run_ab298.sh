#!/bin/bash
# Preflight + the parity set in one: cdk2x2_298 is the parity fixture (cdk2x2_512's soft hinge
# saturates all-atom RMSD for any perturbation), and it also proves every code path before the
# 512 aa timing run commits 10 minutes of card time to it.
WT=/home/ttuser/.coworker/wt/boltz2-diffusion-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=3600 BENCHLOCK_LOAD_WAIT_S=300
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/b2diff/fold_ab_b2diff.py \
    --sizes 298 --arms on,l6,l7l6,s6,l7l6s6,on \
    --out perf/b2diff/ab298_parity.json
echo "RC=$?"
