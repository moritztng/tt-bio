#!/usr/bin/env bash
# One rung of the crop ladder, through of3t-crop640's split_trace.py unchanged.
#
# Usage: run_rung.sh <tokens> <card>
#
# benchlock SERIALISES against other rows' timed measurements; it is not a quiet-box
# requirement. A DRAM high-water is a property of what is resident on one device, so the
# quiet-wait is 0 and the lock only stops a co-tenant's timed A/B running underneath.
set -u
N=$1
CARD=$2
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
export BENCHLOCK_MAXLOAD=99 BENCHLOCK_LOAD_WAIT_S=0 BENCHLOCK_WAIT_S=5400
exec bash /home/ttuser/.coworker/scripts/benchlock.sh worker:of3t-crop768 -- \
  env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" TT_BIO_LEASE_HOLDER=worker:of3t-crop768 \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_crop640/split_trace.py \
    --tokens "$N" --walk-from-gb 18 --walk-step-mb 32 \
    --out perf/of3t_crop768/out/split_"$N".json
