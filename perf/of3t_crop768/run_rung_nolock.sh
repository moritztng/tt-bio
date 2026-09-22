#!/usr/bin/env bash
# One rung of the crop ladder without the benchlock. Usage: run_rung_nolock.sh <tokens> <card>
#
# No lock because this is a BYTE measurement, not a time: a DRAM high-water is a property of
# what is resident on one device and a co-tenant cannot move it. The sibling rung running under
# run_rung.sh holds the benchlock for the same worker, so no other row's timed A/B can start
# underneath either of them while this runs.
set -u
N=$1
CARD=$2
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
exec env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS=0,"$CARD" TT_BIO_LEASE_HOLDER=worker:of3t-crop768 \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_crop640/split_trace.py \
    --tokens "$N" --walk-from-gb 18 --walk-step-mb 32 \
    --out perf/of3t_crop768/out/split_"$N".json
