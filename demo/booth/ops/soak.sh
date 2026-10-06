#!/usr/bin/env bash
# The booth soak: every failure in the ops table injected in rotation against the installed demo,
# then the resource curves over the whole run. Runs detached; follow it in $out/run.log.
#
#   ops/soak.sh [hours]        default 24, one event every 15 min (12 kinds, so each every 3 h)
#
# Writes ~/booth-logs/soak-<UTC stamp>/: events.jsonl and the screenshots (chaos.py), summary.json,
# curves.png and curves.json (curves.py, from the watchdog's log over the same window), screens.txt
# (screens.py: what every sample showed, and every blank, frozen or error one).
# board_reset resets a board under a running fold, so take the chip ledger's go-ahead first.
set -euo pipefail
ops=$(cd "$(dirname "$0")" && pwd)
py=${BOOTH_PYTHON:-$HOME/tt-bio-dev/env/bin/python3}
hours=${1:-24}
since=$(date -u +%Y-%m-%dT%H:%M:%S)
out=$HOME/booth-logs/soak-$(date -u +%m%dT%H%M)
mkdir -p "$out"
events=browser_crash,engine_kill,queue_flood,worker_kill,browser_freeze,network_drop,worker_wedge,sway_freeze,engine_freeze,display_unplug,sway_crash,board_reset
nohup setsid bash -c "
  '$py' -u '$ops/chaos.py' --hours '$hours' --every 900 --events '$events' --out '$out'
  '$py' '$ops/curves.py' ~/booth-logs/watchdog.jsonl --since '$since' --out '$out/curves.png' --json '$out/curves.json'
  '$py' '$ops/screens.py' '$out' > '$out/screens.txt'
" >"$out/run.log" 2>&1 </dev/null &
echo "$!" >"$out/pid"
echo "soak started $since UTC for $hours h: $out (pid $(cat "$out/pid"))"
