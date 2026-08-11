#!/bin/bash
# Shared mutex for exclusive/heavy multi-chip access to the JapanFold Wormhole Galaxy
# (UF-EV-A13-GWH02). Added 2026-08-02 because two coworker-fleet tasks
# (galaxy-32way-scaling-bottleneck and abag-xm-galaxy-sample-scaling-p2) both take device
# windows on this same host with no interlock, and have already collided once (a >2h
# public outage on 2026-08-01 when one worker's sweep held all 32 chips while the other's
# maintenance window was trying to restore service).
#
# Usage:
#   galaxy_device_lock.sh acquire <owner> [wait_seconds=0]   # 0 = do not block, fail fast if busy
#   galaxy_device_lock.sh release <owner>
#   galaxy_device_lock.sh status
#   galaxy_device_lock.sh force-release                      # manual escape hatch only
#
# Rule for both tasks: acquire this BEFORE (a) stopping japanfold.service / opening a
# maintenance window, or (b) running any standalone job that will hold more than ~6-8
# chips concurrently for more than a couple minutes. Release it the MOMENT the heavy
# device phase ends -- do not hold it for your whole pass. If status shows BUSY, do
# host-only/analysis work and retry later rather than contending.

LOCKDIR=~/mthuening/.galaxy_device.lockdir
INFO="$LOCKDIR/owner"
STALE_MIN=60

status() {
  if [ -d "$LOCKDIR" ]; then
    held_by=$(cat "$INFO" 2>/dev/null || echo "(no owner info)")
    if [ -f "$INFO" ]; then
      age_min=$(( ( $(date +%s) - $(stat -c %Y "$INFO") ) / 60 ))
      echo "HELD by $held_by (age ${age_min}m)"
      if [ "$age_min" -ge "$STALE_MIN" ]; then
        echo "STALE (>= ${STALE_MIN}m old) -- if you've confirmed no real job is running (ps aux), force-release is reasonable"
      fi
    else
      echo "HELD (no owner info)"
    fi
  else
    echo "FREE"
  fi
}

cmd="$1"; owner="$2"; wait_s="${3:-0}"

case "$cmd" in
  acquire)
    [ -n "$owner" ] || { echo "usage: acquire <owner> [wait_seconds]"; exit 2; }
    start=$(date +%s)
    while true; do
      if mkdir "$LOCKDIR" 2>/dev/null; then
        echo "$owner @ $(date -u +%FT%TZ)" > "$INFO"
        echo "ACQUIRED by $owner"
        exit 0
      fi
      now=$(date +%s)
      if [ $(( now - start )) -ge "$wait_s" ]; then
        echo "BUSY:"; status
        exit 1
      fi
      sleep 5
    done
    ;;
  release)
    [ -n "$owner" ] || { echo "usage: release <owner>"; exit 2; }
    if [ -f "$INFO" ] && grep -q "^$owner @" "$INFO"; then
      rm -rf "$LOCKDIR"
      echo "RELEASED by $owner"
    else
      echo "REFUSED: lock not held by '$owner' (actual: $(cat "$INFO" 2>/dev/null || echo none))"
      exit 1
    fi
    ;;
  force-release)
    prev=$(cat "$INFO" 2>/dev/null || echo none)
    rm -rf "$LOCKDIR"
    echo "FORCE-RELEASED (was: $prev)"
    ;;
  status|*)
    status
    ;;
esac
