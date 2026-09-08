#!/bin/bash
# Everything still owed, serialized on one chip. Serial because one chip is the partition, not
# because concurrent opens collide -- tenstorrent.py:3499 already serializes bring-up host-wide.
chip=$1
export TT_BIO_LEASE_TIMEOUT=${TT_BIO_LEASE_TIMEOUT:-1800}
T=/home/cust-team/mthuening/ceilpxd/tree
O=/home/cust-team/mthuening/ceilpxd/old
Y=$T/perf/pxdesign/targets
P=/home/cust-team/mthuening/ceilpxd/px_one.sh
bash $P $chip $T $Y/laczc_512.yaml       p512_rep 1 400
bash $P $chip $O $Y/laczc_512.yaml       p512_old 1 400
bash $P $chip $T $Y/laczc_1008_b528.yaml r1536    4 200
bash $P $chip $T $Y/laczc_960.yaml       r1040    4 200
bash $P $chip $T $Y/laczc_1008_b208.yaml r1216    4 200
echo "FINAL CHAIN DONE $(date -u +%H:%M:%S)"
