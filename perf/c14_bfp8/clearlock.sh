#!/usr/bin/env bash
# Clear a stale benchlock owner line. benchlock only truncates $LOCK on a NORMAL exit, so a killed
# holder leaves a line that reads as a live hold while the flock itself is already free -- actively
# misleading to the other three bfp8 rows. The liveness test lives in a script file rather than on
# a command line because an argv-pattern check matches the checking shell itself.
set -u
L="$HOME/.coworker/state/benchlock"
OWNER=$(cat "$L" 2>/dev/null)
case "$OWNER" in
  c14-bfp8-fastpath*) ;;
  *) echo "not mine, leaving alone: ${OWNER:-<empty>}"; exit 0 ;;
esac
PID=$(printf '%s' "$OWNER" | sed -n 's/.*pid=\([0-9]\+\).*/\1/p')
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  echo "owner pid $PID is alive, leaving the lock alone"; exit 0
fi
: > "$L"
echo "cleared stale owner line (pid ${PID:-?} is gone; the flock was already free)"
