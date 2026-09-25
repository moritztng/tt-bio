#!/usr/bin/env bash
# bcx-mutate: a timestamped step-rate curve for an arm, banked every 2 minutes.
#
# BC2's own .campaign_state.json records outcomes but not progress inside a trajectory, and the
# arm log's [pool] lines carry no time. Two multi-hour arms have ended with nothing readable, so
# this writes a CSV a later pass can read to see how far an arm got and how fast, without
# attaching to it.
set -u
ART=/home/ttuser/bcx_mutate_art
SEED=${SEED:?}
LOG=$ART/profile_s${SEED}.log
CSV=$ART/profile_s${SEED}_progress.csv
CARD=${CARD:?}

[ -f "$CSV" ] || echo "utc,pool_swaps,verdicts,claimed,aiclk,load1" > "$CSV"
while :; do
  swaps=$(grep -ac '^\[pool\]' "$LOG" 2>/dev/null || echo 0)
  st=$ART/profile_s${SEED}/.campaign_state.json
  read -r verdicts claimed <<<"$(python3 - "$st" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(sum(d.get("rejections",{}).get("terminated",{}).values()), d.get("trajectories",0))
except Exception:
    print(0,0)
PY
)"
  clk=$(cat /sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk 2>/dev/null || echo NA)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$swaps,$verdicts,$claimed,$clk,$(cut -d' ' -f1 /proc/loadavg)" >> "$CSV"
  sleep 120
done
