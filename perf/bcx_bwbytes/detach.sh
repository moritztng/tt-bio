#!/bin/bash
# Run launch.sh so the job outlives the agent that started it.
#   CARD=<n> detach.sh <out-name> <script.py> <script args...>
#
# Why this exists: the A/B is ~28 rounds plus a JAX compile and a weight load, and a worker turn
# is bounded. A foreground run that does not finish inside one turn is a run that is lost, and
# this row has already spent eighteen turns not measuring anything.
#
# Two rules it exists to honour. The job's cwd is THIS worktree, never a parent or an earlier
# slug's, because a concluded slug's worktree is torn down by fleet hygiene and a live job rooted
# there loses its files mid-run. And `setsid` detaches it from the launching ssh, which is how an
# arm on this fleet died at 23:55 on 2026-09-25: SIGTERM from its own ssh teardown, exit 143.
set -u
: "${CARD:?set CARD to the card being taken -- this row must not guess one}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=$1; shift
LOG=$WT/perf/bcx_bwbytes/runs/$NAME.launch.log
mkdir -p "$(dirname "$LOG")"
cd "$WT" || exit 1
setsid nohup env CARD="$CARD" "$WT/perf/bcx_bwbytes/launch.sh" "$NAME" "$@" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 5
if ! kill -0 "$PID" 2>/dev/null; then
  echo "DETACH FAILED -- launcher exited within 5 s, log follows:" >&2
  cat "$LOG" >&2
  exit 1
fi
echo "detached pid $PID, log $LOG"
