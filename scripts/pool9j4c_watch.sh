#!/bin/bash
# One-off recovery watcher (2026-07-30): the endgame supervisor's pool/9j4c job
# hit abag_xm_frontier_pool.py's 10800 s matrix timeout at 6 workers and its
# auto-retry was killed as doomed-same-sizing. This watcher waits for the A
# labels to drain (frees ~24 workers), rebuilds pool/9j4c at 12 workers with
# the fixed 21600 s timeout, then ensures the analysis+marker step runs
# (relaunching the idempotent endgame supervisor only if it already exited).
set -u
FR=$HOME/abag_xm/frontier
WT=/home/ttuser/.coworker/wt/abag-xm-seeds-vs-samples-oracle-frontier-p2
LOG=$FR/logs/pool9j4c_watch.log

echo "$(date -u +%H:%M:%S) watcher start" >> "$LOG"

# Wait until the 3 A labels exist, or nothing is left working on them
# (supervisor gone and no labels.py procs). Cap 8 h.
for i in $(seq 1 480); do
  if [ -f "$FR/A/9j4c/labels.json" ] && [ -f "$FR/A/9q6y/labels.json" ] && [ -f "$FR/A/9q6z/labels.json" ]; then
    echo "$(date -u +%H:%M:%S) A labels complete" >> "$LOG"; break
  fi
  if ! pgrep -f abag_xm_frontier_endgame >/dev/null && ! pgrep -f scripts/abag_xm_labels.py >/dev/null; then
    echo "$(date -u +%H:%M:%S) supervisor+labels gone, proceeding" >> "$LOG"; break
  fi
  sleep 60
done
sleep 120

if [ ! -f "$FR/B_pool/9j4c/labels.json" ]; then
  echo "$(date -u +%H:%M:%S) building pool/9j4c at 12 workers" >> "$LOG"
  cd "$WT" && POOL_PAIR_WORKERS=12 python3 -u scripts/abag_xm_frontier_pool.py 9j4c \
    >> "$FR/B_pool/9j4c/pool_build.log" 2>&1
  echo "$(date -u +%H:%M:%S) pool/9j4c build rc=$? labels_exists=$([ -f "$FR/B_pool/9j4c/labels.json" ] && echo yes || echo no)" >> "$LOG"
fi

# Ensure the analysis+marker step runs: if the endgame supervisor already
# exited (e.g. it broke out at 11/12 pools), relaunch it — it is idempotent
# and goes straight to analysis when all labels exist.
if [ ! -f "$FR/ENDGAME_DONE" ] && ! pgrep -f abag_xm_frontier_endgame >/dev/null; then
  echo "$(date -u +%H:%M:%S) relaunching endgame for final analysis" >> "$LOG"
  cd "$WT" && setsid nohup python3 -u scripts/abag_xm_frontier_endgame.py </dev/null >/dev/null 2>&1 &
fi
echo "$(date -u +%H:%M:%S) watcher exit" >> "$LOG"
