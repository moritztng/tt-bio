#!/bin/bash
WT=/home/ttuser/.coworker/wt/boltz2-512aa-deep-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-512aa-deep-perf PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=900 BENCHLOCK_LOAD_WAIT_S=300
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-512aa-deep-perf -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py --model boltz2 \
    --sizes 512 --arms on,nos2,sdpa,on --out perf/b2deep/ab_s1s2_512.json
echo "RC=$?"
