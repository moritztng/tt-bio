#!/bin/bash
# wh-perf-esmfold2 p8: the Wormhole roofs, fourth attempt, on UMD 28 (/dev/tenstorrent/4, free per
# `sudo lsof /dev/tenstorrent/[0-7]` at 03:31 UTC). chain9 waited on chain6's PID, which still has
# three sweep legs to go, so this takes a card that is free now instead of one that will be free in
# an hour. chain9 was killed to avoid running the roof twice.
# Attempt 1 lost UMD 30 to a sibling, attempt 2 found UMD 31 wedged (not reset -- production Galaxy).
cd /home/cust-team/mthuening/whbase/wt-esmfold2 || exit 1
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python
export TT_VISIBLE_DEVICES=28
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
echo "=== chain10 done $(date -u +%H:%M:%S) ==="
