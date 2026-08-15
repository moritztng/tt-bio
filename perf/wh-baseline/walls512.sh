#!/bin/sh
# 512 aa per-model wall leg on the Wormhole Galaxy, one model at a time on one card.
#
# 512 matches perf/of3_4xpd/xmodel_qb2c3/ exactly -- same harness, same ttnn 0.68.0, same
# fixture size -- so the WH/BH gap comes out per-model and is not a cross-size comparison.
#
# Waits on the 298 leg by PID. An earlier version waited with `pgrep -f "--size 298"`, which
# also matched the ssh wrapper carrying that string in its own command line, so it waited on
# a process that was never going to exit. Wait on a PID, never on a pattern that can match
# the waiter's own ancestry.
#
# Usage: walls512.sh [pid-to-wait-for]
WAIT_PID="$1"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
T=/home/cust-team/mthuening/whbase/tt-bio
O=/home/cust-team/mthuening/whbase/walls
mkdir -p "$O"
for m in boltz2 protenix-v2 opendde; do
  echo "=== $m 512 start $(date -u +%H:%M:%S) ==="
  TT_VISIBLE_DEVICES=26 TT_METAL_LOGGER_LEVEL=FATAL \
  HF_HUB_CACHE=/home/cust-team/mthuening/whbase/hfcache \
    "$T/env/bin/python" "$T/perf/of3_4xpd/xmodel_ab.py" --model "$m" --tree "$T" --size 512 \
    --repeat 3 --label "wh-$m-512" --out "$O/$m-512.json" > "$O/$m-512.log" 2>&1
  echo "=== $m 512 end rc=$? $(date -u +%H:%M:%S) ==="
done
echo ALLDONE
