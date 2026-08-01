#!/bin/bash
# Safety net for the 32-way scaling sweep window: japanfold.com must never stay on the
# maintenance page because an operator session ended. Holds for MIN_MIN so it cannot fire
# before the window is even set up, then restores as soon as the sweep is done -- or at the
# hard deadline regardless, because a site down indefinitely is worse than a sweep to re-run.
MIN_MIN=${1:-15}
DEADLINE_MIN=${2:-110}
LOG=/home/cust-team/mthuening/g32_restore_watchdog.log
start=$(date +%s); end=$(( start + DEADLINE_MIN*60 )); floor=$(( start + MIN_MIN*60 ))
{
  echo "armed $(date -u) min=${MIN_MIN}min deadline=${DEADLINE_MIN}min"
  while [ "$(date +%s)" -lt "$end" ]; do
    if [ "$(date +%s)" -ge "$floor" ] && ! pgrep -f "[g]32_p2.sh" >/dev/null 2>&1; then
      echo "sweep finished/absent at $(date -u) -> restoring"; break
    fi
    sleep 60
  done
  [ "$(date +%s)" -ge "$end" ] && echo "DEADLINE reached at $(date -u) -> restoring anyway"
  for p in $(ps -eo pid,args | grep "[g]32_p2.sh" | awk "{print \$1}"); do kill -TERM $p; done
  for p in $(ps -eo pid,args | grep "[g]alaxy_conc_sweep" | awk "{print \$1}"); do kill -TERM $p; done
  sleep 20
  for p in $(ps -eo pid,args | grep "[t]t_bio.main predict" | awk "{print \$1}"); do kill -TERM $p; done
  sleep 40
  bash /home/cust-team/mthuening/maintenance/maint-restore.sh
  echo "restore launched $(date -u)"
} >> "$LOG" 2>&1
