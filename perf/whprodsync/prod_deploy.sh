#!/bin/bash
# Sync JapanFold prod to tt-bio origin/main (four ESMFold2 levers, verified on Wormhole).
# v2: drain check reads pgrep correctly, and the tree fast-forward happens before the stop so the
# outage is stop->start only. Watchdog timer stays paused for the whole window: it fires every
# 2 min and issues `systemctl restart`, which is exactly the dirty restart the wedge memory forbids.
set -u
TREE=/home/cust-team/mthuening/aiand-bio
holders(){ sudo lsof -t /dev/tenstorrent/* 2>/dev/null | sort -u | wc -l | tr -d ' '; }
procs(){ pgrep -f "[t]t-bio worker|[t]t-bio serve" 2>/dev/null | wc -l | tr -d ' '; }
log(){ echo "$(date -u +%H:%M:%S)Z $*"; }

sudo systemctl stop japanfold-watchdog.timer
log "watchdog timer: $(systemctl is-active japanfold-watchdog.timer)"
log "PRE holders=$(holders) procs=$(procs) jobs=$(curl -s -m 5 http://127.0.0.1:8090/api/jobs)"
log "PRE tree: $(git -C $TREE log --oneline -1)"

git -C $TREE fetch -q origin
git -C $TREE merge --ff-only origin/main 2>&1 | tail -2
log "POST tree: $(git -C $TREE log --oneline -1)"
log "dirty files: $(git -C $TREE status --porcelain | wc -l)"

STOPPED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sudo systemctl stop japanfold.service
log "stopped, STOPPED_AT=$STOPPED_AT"

OK=0
for i in $(seq 1 40); do
  H=$(holders); P=$(procs)
  log "drain $i holders=[$H] procs=[$P]"
  if [ "$H" -eq 0 ] && [ "$P" -eq 0 ]; then OK=1; break; fi
  sleep 3
done
if [ "$OK" != "1" ]; then
  log "survivors after 120s, killing leftovers by explicit pid (unit already stopped)"
  for p in $(pgrep -f "[t]t-bio worker|[t]t-bio serve"); do log "  kill -9 $p"; sudo kill -9 "$p" 2>/dev/null; done
  sleep 10
  H=$(holders); P=$(procs); log "after kill holders=[$H] procs=[$P]"
  if [ "$H" -ne 0 ] || [ "$P" -ne 0 ]; then
    log "ABORT: chips still held, not starting on top of survivors"
    sudo systemctl start japanfold-watchdog.timer
    exit 9
  fi
fi

sudo systemctl start japanfold.service
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
log "STARTED_AT=$STARTED_AT"
for i in $(seq 1 60); do
  R=$(curl -s -m 5 http://127.0.0.1:8090/api/health 2>/dev/null)
  log "health $i: $R"
  case "$R" in *'"status":"ok"'*) break;; esac
  sleep 5
done
log "holders after start: $(holders)"
log "serving tree: $(git -C $TREE log --oneline -1)"
python3 - "$STOPPED_AT" "$STARTED_AT" <<'PY'
import sys, datetime
f="%Y-%m-%dT%H:%M:%SZ"
a=datetime.datetime.strptime(sys.argv[1],f); b=datetime.datetime.strptime(sys.argv[2],f)
print("STOPPED_AT=%s" % sys.argv[1]); print("STARTED_AT=%s" % sys.argv[2])
print("OUTAGE_SECONDS=%d" % (b-a).total_seconds())
PY
sudo systemctl start japanfold-watchdog.timer
log "watchdog timer restored: $(systemctl is-active japanfold-watchdog.timer)"
log "DEPLOY_SCRIPT_DONE"
