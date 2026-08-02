#!/bin/bash
# Restore watchdog for the p25 galaxy window (deep-N saturation, control + pilot).
# Pattern copied from p24_watchdog.sh (proven): restore when the fleet finishes, or at the
# deadline regardless. Run pattern HARDCODED with bracket-glob so pgrep never matches this
# script's own command line.
RUNPAT='[p]25_fleet.sh'
MIN_MIN=20
DEADLINE_MIN=270
LOG=/home/cust-team/mthuening/p25_watchdog.log
start=$(date +%s); end=$(( start + DEADLINE_MIN*60 )); floor=$(( start + MIN_MIN*60 ))
{
  echo "armed $(date -u) deadline=${DEADLINE_MIN}min"
  while [ "$(date +%s)" -lt "$end" ]; do
    if [ "$(date +%s)" -ge "$floor" ] && ! pgrep -f "$RUNPAT" >/dev/null 2>&1; then
      echo "run finished at $(date -u) -> restoring"; break
    fi
    sleep 30
  done
  [ "$(date +%s)" -ge "$end" ] && echo "DEADLINE at $(date -u) -> restoring anyway"

  for p in $(pgrep -f "$RUNPAT"); do
    pgid=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && kill -TERM -"$pgid" 2>/dev/null
  done
  sleep 20
  # Sweep leftovers holding a chip. SIGTERM only -- a kill -9 leaves the chip dirty and the
  # damage only shows at the NEXT open.
  for p in $(/usr/bin/python3.10 -c "
import os,glob
out=set()
for f in glob.glob('/proc/[0-9]*/fd/*'):
    try: t=os.readlink(f)
    except OSError: continue
    if not t.startswith('/dev/tenstorrent/'): continue
    pid=f.split('/')[2]
    try: c=open('/proc/'+pid+'/cmdline').read().replace(chr(0),' ')
    except OSError: continue
    if '/mthuening/tt-bio/env/' not in c: out.add(pid)
print(' '.join(sorted(out)))
"); do kill -TERM "$p" 2>/dev/null; done
  # venv-python fleet folds (esmfold2 control) share the service's interpreter path; they die
  # with the RUNPAT process-group TERM above. Belt: TERM any surviving p25 fold processes by
  # their out-dir marker.
  for p in $(pgrep -f 'mthuening/p25/'); do kill -TERM "$p" 2>/dev/null; done
  sleep 30

  bash /home/cust-team/mthuening/maintenance/maint-restore.sh
  echo "restore launched $(date -u)"
  sleep 240
  for i in 1 2 3 4 5 6; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 https://japanfold.com || echo 000)
    echo "  external check $i: HTTP $code"
    [ "$code" = "200" ] && break
    sleep 30
  done
  if [ "${code:-000}" != "200" ]; then
    echo "forcing tunnel back to :8090"
    sed -i 's|service: http://localhost:8091|service: http://localhost:8090|g' /home/cust-team/.cloudflared/config.yml
    sudo systemctl restart cloudflared-japanfold.service
    sleep 20
    echo "  after forced flip: HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 30 https://japanfold.com)"
  fi
  bash /home/cust-team/mthuening/galaxy_device_lock.sh release worker:abag-xm-deepn-saturation-fullpanel \
    || bash /home/cust-team/mthuening/galaxy_device_lock.sh force-release
  echo "done $(date -u)"
} >> "$LOG" 2>&1
