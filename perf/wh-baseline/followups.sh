#!/bin/sh
# The three measurements still owed after the 512 aa wall leg, run sequentially on one card so
# no two timed runs ever overlap on this shared host.
#
#   1. reblock_permute at C=64. The C=128 sweep says N=256 on an L1 destination wins 1.617x on
#      Wormhole where the shipped gate declines it citing 0.952x on Blackhole. BoltzGen's dominant
#      channel move is [1,256,256,64], i.e. C=64, so the number that would decide the window's
#      lower edge has to be measured at C=64 before it is quoted as a BoltzGen win.
#   2. protenix-v2 at 298 aa with stderr kept. Its first run crashed and the wrapper's `tail -20`
#      threw away the exception line, so the cause is still unknown.
#   3. esmfold2 at 512 aa in --fast. Forced on Wormhole (ESMC-6B is ~12.8 GB, the chip has ~12),
#      and --fast is only comparable to another --fast arm, which Blackhole does not have yet.
#
# Usage: followups.sh [pid-to-wait-for]
WAIT_PID="$1"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
T=/home/cust-team/mthuening/whbase/tt-bio
O=/home/cust-team/mthuening/whbase/walls
W=/home/cust-team/mthuening/whbase
mkdir -p "$O"
export TT_VISIBLE_DEVICES=26
export TT_METAL_LOGGER_LEVEL=FATAL
export HF_HUB_CACHE=/home/cust-team/mthuening/whbase/hfcache

echo "=== reblock C=64 start $(date -u +%H:%M:%S) ==="
PROBE_C=64 PYTHONPATH="$T" "$T/env/bin/python" "$W/wh_reblock_window.py" > "$W/reblock_c64.log" 2>&1
echo "=== reblock C=64 end rc=$? $(date -u +%H:%M:%S) ==="

echo "=== protenix-v2 298 repro start $(date -u +%H:%M:%S) ==="
"$T/env/bin/python" "$T/perf/of3_4xpd/xmodel_ab.py" --model protenix-v2 --tree "$T" --size 298 \
  --repeat 3 --label wh-protenix-v2 --out "$O/protenix-v2-298.json" > "$O/protenix-v2-298.log" 2>&1
echo "=== protenix-v2 298 end rc=$? $(date -u +%H:%M:%S) ==="

echo "=== esmfold2 512 fast start $(date -u +%H:%M:%S) ==="
"$T/env/bin/python" "$T/perf/of3_4xpd/xmodel_ab.py" --model esmfold2 --tree "$T" --size 512 --fast \
  --repeat 3 --label wh-esmfold2-512-fast --out "$O/esmfold2-512-fast.json" \
  > "$O/esmfold2-512-fast.log" 2>&1
echo "=== esmfold2 512 fast end rc=$? $(date -u +%H:%M:%S) ==="
echo FOLLOWUPS_DONE
