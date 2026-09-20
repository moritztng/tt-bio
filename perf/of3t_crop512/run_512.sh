#!/usr/bin/env bash
# The 512 rung, run through of3t-crop640's split_trace.py unchanged.
#
# benchlock is taken to SERIALISE against other rows' timed measurements -- not because this one
# needs a quiet box. A DRAM high-water is a property of what is resident on one device, so the
# quiet-wait is set to 0 and the lock is held only so a co-tenant's timed A/B does not run
# underneath this. BENCHLOCK_MAXLOAD=99 + LOAD_WAIT_S=0 means "acquire, then start"; the warning
# it prints about proceeding under load is expected and is not a defect in this measurement.
set -u
WT=/home/ttuser/.coworker/wt/of3t-crop512
cd "$WT"
export BENCHLOCK_MAXLOAD=99 BENCHLOCK_LOAD_WAIT_S=0 BENCHLOCK_WAIT_S=5400
exec bash /home/ttuser/.coworker/scripts/benchlock.sh worker:of3t-crop512 -- \
  env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:of3t-crop512 \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_crop640/split_trace.py \
    --tokens 512 --walk-from-gb 18 --walk-step-mb 32 \
    --out perf/of3t_crop512/out/split_512.json
