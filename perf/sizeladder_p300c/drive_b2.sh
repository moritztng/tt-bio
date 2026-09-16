#!/bin/bash
# Record one boltz-2 p300c slice, relaunching it when a fold WEDGES instead of waiting it out.
# A wedged boltz-2 fold on qb2 does not fail: the process sits at 0% CPU with no further log line
# and the slice burns the whole uptime window for nothing (four sightings in pass 6, three cards,
# two rungs; it reads as the per-card PCIe link failure QBROOT closed on, seen from tt-bio). The
# trio needs ~6 min, so no CPU progress across the fold tree for 150 s means dead, not slow.
# $1 = card, $2 = rungs, $3 = per-try seconds (default 1200), $4 = tries (default 8).
CARD=$1; RUNGS=$2; LIMIT=${3:-1200}; TRIES=${4:-8}
WT=/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
cd "$WT" || exit 1
HEAD=$(git rev-parse --short HEAD)
have() {  # every requested rung present AND stamped at the current HEAD
  /home/ttuser/tt-bio-dev/env/bin/python3 - "$RUNGS" "$HEAD" <<'PY'
import json, sys
rungs, head = sys.argv[1].split(","), sys.argv[2]
try:
    e = json.load(open("docs/size_ladder_baseline.d/boltz2.json"))["cards"]["p300c"]["models"]["boltz2"]
except Exception:
    sys.exit(1)
sys.exit(0 if e.get("commit") == head and all(r in e.get("runtime_s", {}) for r in rungs) else 1)
PY
}
kill_tree() {
  for p in $(pgrep -f "size-ladder-record") $(pgrep -f "tt_bio.main predict") \
           $(pgrep -f "lever_census.py") $(pgrep -f "spawn_main"); do kill -9 "$p" 2>/dev/null; done
  sleep 6
}
for try in $(seq 1 "$TRIES"); do
  if have; then echo "ALREADY HAVE ${RUNGS} at ${HEAD}"; exit 0; fi
  kill_tree
  LOG="perf/sizeladder_p300c/rec_b2_${RUNGS//,/_}_t${try}.log"
  setsid nohup bash perf/sizeladder_p300c/rec_boltz2_part.sh "$CARD" "$RUNGS" >"$LOG" 2>&1 </dev/null &
  sleep 8
  PID=$(pgrep -f "size-ladder-record" | head -1)
  echo "$(date -u +%H:%M:%S) try $try pid ${PID:-none} -> $LOG"
  [ -z "$PID" ] && continue
  last=-1; stall=0; t=0
  while kill -0 "$PID" 2>/dev/null && [ "$t" -lt "$LIMIT" ]; do
    sleep 15; t=$((t+15))
    cur=$(ps -eo cputimes,args | grep -E "tt_bio\.main|lever_census" | grep -v grep \
          | awk '{s+=$1} END {print s+0}')
    if [ "$cur" -le "$last" ]; then stall=$((stall+15)); else stall=0; fi
    last=$cur
    if [ "$stall" -ge 150 ]; then
      echo "$(date -u +%H:%M:%S) WEDGE: fold tree burned no CPU for ${stall}s at t=${t}s, relaunching"
      kill_tree; break
    fi
  done
  kill -0 "$PID" 2>/dev/null && { echo "$(date -u +%H:%M:%S) over limit ${LIMIT}s"; kill_tree; }
  if have; then echo "$(date -u +%H:%M:%S) RECORDED ${RUNGS} at ${HEAD} on try $try"; exit 0; fi
  echo "$(date -u +%H:%M:%S) try $try produced nothing"
  git checkout -- docs/size_ladder_baseline.json 2>/dev/null
done
echo "GAVE UP after $TRIES tries"; exit 1
