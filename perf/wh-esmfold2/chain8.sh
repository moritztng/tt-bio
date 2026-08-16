#!/bin/bash
# wh-perf-esmfold2 p6: the 1024 aa leg, retried on its own card. The chain6 attempt died on
# DeviceInUseError against a sibling task holding UMD 30 -- 1024 is JapanFold's max_residues, so
# this is the one size the sweep cannot be missing. UMD 28 = /dev/tenstorrent/4, unoccupied per
# `sudo lsof /dev/tenstorrent/[0-7]` at 02:49 UTC.
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
echo "=== sweep esmfold2 1024 RETRY start $(date -u +%H:%M:%S) ==="
$BL wh-perf-esmfold2 -- $PY -u perf/wh-esmfold2/fold_ab512.py \
    --model esmfold2 --size 1024 --fast --arms base,A --rounds 1 \
    --out "$O/sweep_esmfold2_1024_wh.json" > "$O/sweep_esmfold2_1024_wh.log" 2>&1
echo "=== sweep esmfold2 1024 RETRY rc=$? $(date -u +%H:%M:%S) ==="
echo "=== chain8 done $(date -u +%H:%M:%S) ==="
