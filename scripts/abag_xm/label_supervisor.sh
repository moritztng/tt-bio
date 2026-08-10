#!/bin/bash
# Restarts the deep-N labeler if its python dies, for the unattended overnight window.
#
# Detection matches the PYTHON process only. The `bash -c` wrapper that launched the
# labeler carries the same script name in its own cmdline and outlives the python, so a
# bare `pgrep -f abag_xm_deepn_label.py` is a permanent false negative: it keeps matching
# the wrapper after the real labeler is gone, and the supervisor never fires. For the same
# reason never `pkill -f label_supervisor.sh` -- that pattern matches the shell you are
# typing it into.
#
# The labeler itself never exits on drain (it needs galaxy/CAMPAIGN_DONE), so this
# supervisor only guards against a silent death; it does not detect completion.
set -u
WT=$HOME/.coworker/wt/abag-n512
BASE=$HOME/abag_xm/deepn/galaxy
LOG=$HOME/abag_xm/deepn/logs/label_n512.log
SLOG=$HOME/abag_xm/deepn/logs/label_supervisor.log

alive() {
  ps -eo comm= -o args= |
    awk '$1=="python3" && /abag_xm_deepn_label\.py/ {n++} END {exit !(n>0)}'
}

exec 9>/tmp/.abag_label_sup.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another supervisor holds the lock, exiting" >>"$SLOG"; exit 0; }
echo "$(date -u +%FT%TZ) supervisor up (pid $$), labeler alive=$(alive && echo yes || echo no)" >>"$SLOG"

while true; do
  if ! alive; then
    echo "$(date -u +%FT%TZ) labeler python ABSENT -- relaunching" >>"$SLOG"
    (cd "$WT" && setsid nohup nice -15 python3 -u scripts/abag_xm_deepn_label.py \
       --base "$BASE" --workers 12 </dev/null >>"$LOG" 2>&1 &)
    sleep 30
    if alive; then
      echo "$(date -u +%FT%TZ) relaunch OK" >>"$SLOG"
    else
      echo "$(date -u +%FT%TZ) relaunch FAILED" >>"$SLOG"
    fi
  fi
  sleep 120
done
