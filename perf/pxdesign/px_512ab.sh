#!/bin/bash
# The 512 residual on this engine vs the engine the catalog recorded it on. Same chip, same
# fixture, same flags; the only variable is the tt_bio checkout.
chip=$1
Y=/home/cust-team/mthuening/ceilpxd/tree/perf/pxdesign/targets/laczc_512.yaml
export TT_BIO_LEASE_TIMEOUT=900
bash /home/cust-team/mthuening/ceilpxd/px_one.sh $chip /home/cust-team/mthuening/ceilpxd/tree $Y p512_rep 1 400
bash /home/cust-team/mthuening/ceilpxd/px_one.sh $chip /home/cust-team/mthuening/ceilpxd/old  $Y p512_old 1 400
echo "512 A/B DONE $(date -u +%H:%M:%S)"
