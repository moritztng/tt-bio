#!/bin/bash
# wh-perf-esmfold2 p6: the Wormhole roofs, bounded. Every leg is wrapped in `timeout` because the
# p2 roofs.py run held the shared benchlock for 2 h 20 m on a device op that never returned, and a
# hung ttnn call cannot be interrupted from inside Python.
cd /home/cust-team/mthuening/whbase/wt-esmfold2 || exit 1
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python
export TT_VISIBLE_DEVICES=30
export TT_BIO_LEASE_HOLDER=worker:wh-perf-esmfold2
export TT_METAL_LOGGER_LEVEL=FATAL
export BENCHLOCK_FILE=/home/cust-team/mthuening/whbase/benchlock
export BENCHLOCK_FOREIGN_RE="xmodel_ab|wh_esm|decomp\.py|fold_ab|screen_wh|roofs\.py|roof_wh"
export BENCHLOCK_MAXLOAD=20
export BENCHLOCK_LOAD_WAIT_S=120
BL=/home/cust-team/mthuening/whbase/benchlock.sh
O=perf/wh-esmfold2/out
echo "=== roof_wh start $(date -u +%H:%M:%S) ==="
$BL wh-perf-esmfold2 -- timeout -s KILL 900 $PY -u perf/wh-esmfold2/roof_wh.py \
    --L 512 --fast --out "$O/roof_512_wh.json" > "$O/roof_512_wh.log" 2>&1
echo "=== roof_wh rc=$? $(date -u +%H:%M:%S) ==="
echo "=== chain7 done $(date -u +%H:%M:%S) ==="
