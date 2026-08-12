#!/bin/bash
# L2 (scale fold) + L1 (AdaLN s-norm hoist) + the owed S2 fold-level replicate, one process.
# 12 folds: on x3 (the A/A floor), nos2 x3 (S2 control), s3 x2, l1 x2, both x2, order alternated so
# a monotonic drift cannot masquerade as a lever.
WT=/home/ttuser/.coworker/wt/boltz2-512aa-deep-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-512aa-deep-perf PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=300
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-512aa-deep-perf -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py --model boltz2 \
    --sizes 512 --arms on,s3,l1,both,nos2,on,nos2,both,l1,s3,on,nos2 \
    --out perf/b2deep/ab_l1l2_512.json
echo "RC=$?"
