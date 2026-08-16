#!/bin/bash
# wh-perf-esmfold2 p6: the roof, third attempt, chained on chain6's PID rather than on a pattern.
#
# Attempt 1 (UMD 30) died on DeviceInUseError against a sibling task. Attempt 2 (UMD 31) died on
# "Device 0 init: failed to initialize FW! Try resetting the board" -- UMD 31 is the card whose
# roofs.py was SIGKILL'd mid-device-op in p4, and it has not recovered. It is NOT reset here: board
# resets are forbidden on this production Galaxy, and the card is left alone and reported instead.
#
# Every free card on the box is now spoken for by one of this task's own chains, so waiting for
# chain6 to exit is the only way to name a card that will still be free when the roof opens it.
# benchlock serialises the timed run; the device lease serialises the open; this waits for both.
cd /home/cust-team/mthuening/whbase/wt-esmfold2 || exit 1
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python
CHAIN6_PID=3943567
while kill -0 "$CHAIN6_PID" 2>/dev/null; do sleep 30; done
echo "=== chain6 exited, roof may take UMD 30 $(date -u +%H:%M:%S) ==="
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
echo "=== chain9 done $(date -u +%H:%M:%S) ==="
