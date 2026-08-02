#!/bin/bash
# Safety net for a scaling-sweep maintenance window: japanfold.com must never stay on the
# maintenance page because an operator session ended. Holds for MIN_MIN so it cannot fire before
# the window is even set up, then restores when the sweep is done -- or at the hard deadline
# regardless, because a site left down indefinitely is worse than a sweep that has to be re-run.
#
#   galaxy_maint_window_watchdog.sh [min-minutes] [deadline-minutes] [sweep-pattern]
#
# It only ever ends the window it was armed for. The first version of this ended ANY window and
# SIGTERMed EVERY `tt_bio.main predict` on the box, which would have killed another worker's
# campaign mid-batch had it fired at the wrong moment -- on a shared box "restore production" is
# not automatically a safe thing to do. So it fingerprints the maintenance HOLD at arm time and
# stays out of the way if the hold it finds later is somebody else's, and it kills only the sweep
# it was told about, never folds it does not own.
MIN_MIN=${1:-15}
DEADLINE_MIN=${2:-110}
SWEEP_PAT=${3:-g32_p2.sh}
LOG=/home/cust-team/mthuening/g32_restore_watchdog.log
HOLD=/home/cust-team/.japanfold-agent/MAINTENANCE-HOLD.md

start=$(date +%s); end=$(( start + DEADLINE_MIN*60 )); floor=$(( start + MIN_MIN*60 ))
mine=$(grep -E "^Opened:" "$HOLD" 2>/dev/null)

{
  echo "armed $(date -u) min=${MIN_MIN}min deadline=${DEADLINE_MIN}min sweep='${SWEEP_PAT}'"
  echo "  window fingerprint: ${mine:-<none yet; will re-read at fire time>}"
  while [ "$(date +%s)" -lt "$end" ]; do
    if [ -z "$mine" ]; then mine=$(grep -E "^Opened:" "$HOLD" 2>/dev/null); fi
    if [ "$(date +%s)" -ge "$floor" ] && ! pgrep -f "[${SWEEP_PAT:0:1}]${SWEEP_PAT:1}" >/dev/null 2>&1; then
      echo "sweep finished/absent at $(date -u)"; break
    fi
    sleep 60
  done
  [ "$(date +%s)" -ge "$end" ] && echo "DEADLINE reached at $(date -u)"

  now=$(grep -E "^Opened:" "$HOLD" 2>/dev/null)
  if [ ! -f "$HOLD" ]; then
    echo "no maintenance hold in place -- nothing to restore, standing down"; exit 0
  fi
  if [ -n "$mine" ] && [ "$now" != "$mine" ]; then
    echo "the open window is NOT the one I was armed for -- standing down"
    echo "  armed for: $mine"
    echo "  found:     $now"
    exit 0
  fi

  # only the sweep this watchdog was armed for, never folds it does not own
  for p in $(ps -eo pid,args | grep "[${SWEEP_PAT:0:1}]${SWEEP_PAT:1}" | awk '{print $1}'); do
    kill -TERM "$p" 2>/dev/null
  done
  sleep 20
  for p in $(ps -eo pid,args | grep "[t]t_bio.main worker --connect http://127.0.0.1:8899" | awk '{print $1}'); do
    kill -TERM "$p" 2>/dev/null
  done
  for p in $(ps -eo pid,args | grep "[t]t_bio.main controller --listen 127.0.0.1:8899" | awk '{print $1}'); do
    kill -TERM "$p" 2>/dev/null
  done
  sleep 40
  bash /home/cust-team/mthuening/maintenance/maint-restore.sh
  echo "restore launched $(date -u)"
} >> "$LOG" 2>&1
