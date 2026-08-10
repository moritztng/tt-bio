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
#
# Restarting is not enough on its own. `label_one` writes `<fold>/.label_lock` and unlinks it
# in a `finally`, which a SIGKILL never runs, and `pending_folds` skips any fold carrying a
# lock BEFORE it looks at anything else. So a hard kill of a 12-worker labeler leaves 12 locked
# folds that every later scan skips forever: those cells never reach 8/8 chunks,
# verify_n512_nesting.py SKIPS partial cells rather than failing on them, and the 512 panel
# silently comes out up to 12 cells short. reap_locks() removes a lock whose recorded pid is
# gone, which is every lock at the moment a relaunch is due.
set -u
WT=$HOME/.coworker/wt/abag-n512
BASE=$HOME/abag_xm/deepn/galaxy
LOG=$HOME/abag_xm/deepn/logs/label_n512.log
SLOG=$HOME/abag_xm/deepn/logs/label_supervisor.log
HEARTBEAT_EVERY=15   # loop iterations of 120 s, so ~30 min

alive() {
  ps -eo comm= -o args= |
    awk '$1=="python3" && /abag_xm_deepn_label\.py/ {n++} END {exit !(n>0)}'
}

reap_locks() {
  local n=0 pid
  while IFS= read -r lk; do
    pid=$(cat "$lk" 2>/dev/null)
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$lk" && n=$((n + 1))
    fi
  done < <(find "$BASE" -mindepth 3 -maxdepth 3 -name .label_lock 2>/dev/null)
  echo "$n"
}

# Every child gets `9>&-`. fd 9 is the singleton lock and a child inherits it, which cost a
# supervisor swap once: killing the supervisor left its orphaned `sleep 120` holding the lock,
# so the replacement read "another supervisor holds the lock" and exited, leaving the labeler
# unsupervised. The relaunched labeler is worse -- it lives for hours, so it would lock out
# every future supervisor for its whole run.
exec 9>/tmp/.abag_label_sup.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another supervisor holds the lock, exiting" >>"$SLOG"; exit 0; }
echo "$(date -u +%FT%TZ) supervisor up (pid $$), labeler alive=$(alive && echo yes || echo no)" >>"$SLOG"

i=0
while true; do
  if ! alive; then
    echo "$(date -u +%FT%TZ) labeler python ABSENT -- reaped $(reap_locks) stale lock(s), relaunching" >>"$SLOG"
    (cd "$WT" && setsid nohup nice -15 python3 -u scripts/abag_xm_deepn_label.py \
       --base "$BASE" --workers 12 </dev/null >>"$LOG" 2>&1 9>&- &)
    sleep 30 9>&-
    if alive; then
      echo "$(date -u +%FT%TZ) relaunch OK" >>"$SLOG"
    else
      echo "$(date -u +%FT%TZ) relaunch FAILED" >>"$SLOG"
    fi
    i=0
  fi
  # Timestamped progress, so the overnight rate is readable from one file instead of two
  # manual samples minutes apart.
  i=$((i + 1))
  if [ $((i % HEARTBEAT_EVERY)) -eq 0 ]; then
    echo "$(date -u +%FT%TZ) heartbeat: ok=$(grep -c ' ok ' "$LOG" 2>/dev/null) FAILED=$(grep -c FAILED "$LOG" 2>/dev/null)" >>"$SLOG"
  fi
  sleep 120 9>&-
done
