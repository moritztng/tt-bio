#!/usr/bin/env bash
# One rung of the crop ladder. Usage: run_rung2.sh <tokens> <card> <on|off>
#
# The third argument is the dead-value release arm, and it lands in the artifact rather than
# being inferred from a commit hash later. No benchlock: this is a BYTE measurement, not a
# time, and a DRAM high-water is a property of what is resident on one device.
set -u
N=$1; CARD=$2; ARM=$3
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
exec env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS=0,"$CARD" TT_BIO_LEASE_HOLDER=worker:of3t-crop768 \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_crop640/split_trace.py \
    --tokens "$N" --walk-from-gb 18 --walk-step-mb 32 --dead-values "$ARM" \
    --out perf/of3t_crop768/out/split_"$N"_"$ARM".json
