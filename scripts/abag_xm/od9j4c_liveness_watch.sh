#!/bin/bash
# Persistent recorder for od9j4c_liveness.sh. Runs on the Galaxy under setsid, appends a stamped
# liveness block every INTERVAL seconds to p34d/liveness.log, and exits once every chunk is DONE.
#
# The point is that the next pass reads a HISTORY, not a single instantaneous check. Two passes of
# this campaign steered by a predicted schedule and lost hours to a death nobody was watching for;
# a stalled chunk shows up here as the same last_line repeating, timestamped, without anyone having
# to be awake for it.
#
# Singleton by pidfile, not by pgrep. `pgrep -f od9j4c_liveness_watch` matches the very ssh command
# that launches it, so a pgrep guard reports ALREADY_RUNNING on a box where nothing is running.
set -u
M=${M:-$HOME/mthuening}
PF=$M/p34d/liveness_watch.pid
LOG=$M/p34d/liveness.log
INTERVAL=${INTERVAL:-300}

if [ -r "$PF" ] && kill -0 "$(cat "$PF")" 2>/dev/null && [ "$(cat "$PF")" != "$$" ]; then
  echo "already running: pid $(cat "$PF")" >&2; exit 3
fi
echo $$ > "$PF"
trap 'rm -f "$PF"' EXIT

while :; do
  {
    date -u +%Y-%m-%dT%H:%M:%SZ
    out=$(bash "$M/od9j4c_liveness.sh"); rc=$?
    printf '%s\nrc=%s\n\n' "$out" "$rc"
  } >> "$LOG" 2>&1
  # Stop when the whole cell is folded: eight chunks, all DONE.
  if [ "$(printf '%s\n' "$out" | grep -c 'DONE$')" -ge 8 ]; then
    printf 'all 8 chunks DONE, watcher exiting\n' >> "$LOG"; exit 0
  fi
  sleep "$INTERVAL"
done
