#!/bin/bash
# One-shot p29 harvest sweep-2 relauncher (pass-329 plan): wait for the sweep-1
# harvest pid to exit, then relaunch the identical skip-if-complete sweep once.
# Defers to the drain sentinel (its state file) and to any already-running p29
# harvest (one-survivor rule, pass-303 lesson). Setsid-detached; 14h hard cap.
set -u
SWEEP1_PID=${1:?usage: p29_harvest_sweep2_watch.sh <sweep1_pid>}
STATE=$HOME/.coworker/state/deepn_harvested_p29
WT=$HOME/.coworker/wt/abag-xm-deepn-saturation-fullpanel
LOG=/tmp/p29_harvest2.log
end=$(( $(date +%s) + 50400 ))
while kill -0 "$SWEEP1_PID" 2>/dev/null; do
  sleep 60
  [ "$(date +%s)" -ge "$end" ] && exit 0
done
sleep 30   # let sweep 1's embedded python finish its last rsync
[ -f "$STATE" ] && exit 0
pgrep -f "p25_harvest.s[h] p29" >/dev/null && exit 0
cd "$WT" && DEST=$HOME/qb1_galaxy setsid nohup bash scripts/abag_xm/p25_harvest.sh p29 256 >> "$LOG" 2>&1 &
echo "$(date -u +%FT%TZ) sweep 2 launched pid=$!" >> "$LOG"
