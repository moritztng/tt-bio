#!/bin/bash
# dfd-perf: move the wls4 measurement tree to c1a36ab18 (one-slice x1 gather) only if the card
# A/B on anx-glx says that arm is bit-identical to base, and only before wls4 has started.
T=/home/ttuser/.coworker/wt/dfd-perf/artifacts/dfd-perf/headmeas; L=/home/ttuser/scratch/dfd/perf/pin_head.log
end=$(date -ud 2026-11-28T14:30:00Z +%s)
while [ $(date +%s) -lt $end ]; do
  grep -q " start chip" /home/ttuser/scratch/dfd/perf/wls4.log && { echo "wls4 already started, tree left at $(git -C $T rev-parse --short HEAD) $(date -u +%FT%TZ)" >> $L; exit 0; }
  v=$(timeout 20 ssh -o BatchMode=yes anx-glx 'cat ~/dfd/out/ab1_tile.json 2>/dev/null' | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["arms"]["x1"]["vs_base"]["bit_identical"])
except Exception: print("none")')
  if [ "$v" = True ]; then
    git -C $T fetch -q origin wk/dfd-perf 2>/dev/null; git -C $T checkout -q c1a36ab18 && echo "x1 bit-identical on card, tree at $(git -C $T rev-parse --short HEAD) $(date -u +%FT%TZ)" >> $L
    exit 0
  fi
  [ "$v" = False ] && { echo "x1 NOT bit-identical, tree left $(date -u +%FT%TZ)" >> $L; exit 0; }
  sleep 60
done
