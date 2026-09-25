#!/bin/sh
# External MemAvailable floor for a run whose own harness has none. bwprof.py is
# of3t-bwattrib's and is imported, never edited, so the guard goes beside it rather than in
# it. Kills the SUBJECT by explicit pid -- never pkill -- so the kernel OOM-killer never gets
# to choose a victim by score, which on 2026-09-25 picked of3t-restep's step.
PID=$1; FLOOR_KB=$2; LOG=$3
while kill -0 "$PID" 2>/dev/null; do
  A=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  if [ "$A" -lt "$FLOOR_KB" ]; then
    echo "GUARD: MemAvailable ${A} kB under floor ${FLOOR_KB} kB -- killing subject pid $PID (SIGTERM)" >> "$LOG"
    kill -TERM "$PID"
    sleep 20
    kill -0 "$PID" 2>/dev/null && { echo "GUARD: still alive, SIGKILL $PID" >> "$LOG"; kill -KILL "$PID"; }
    exit 37
  fi
  sleep 2
done
echo "GUARD: subject $PID exited on its own, no intervention" >> "$LOG"
