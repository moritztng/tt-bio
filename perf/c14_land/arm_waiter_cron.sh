#!/usr/bin/env bash
# Arm the measurement waiter so it survives a qb2 hard reset. Idempotent: re-running only
# refreshes the deadline and re-installs the one crontab line.
#
# Usage: bash perf/c14_land/arm_waiter_cron.sh [hours]   (default 12)
set -eu
WT=/home/ttuser/.coworker/wt/c14-land-tail
MARK=c14-land-tail-waiter
HOURS=${1:-12}
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
echo "$DEADLINE" > "$WT/perf/c14_land/waiter_deadline"

# Every 3 minutes covers the reboot case too: cron comes back with the box, so the first tick
# after a reset re-arms the waiter without anyone being awake to do it.
LINE="*/3 * * * * CARD=1 bash $WT/perf/c14_land/waiter_tick.sh   # $MARK"

# flock, not a bare read-modify-write: ttuser's crontab is shared with other rows (train-i-run
# has an @reboot line in it) and two rows installing at once would drop one of them.
( flock 9
  { crontab -l 2>/dev/null | grep -v "$MARK"; echo "$LINE"; } | crontab -
) 9>/tmp/c14_crontab.lock

echo "ARMED until $(date -Is -d "@$DEADLINE")"
crontab -l | grep -c . | sed 's/^/crontab lines now: /'
crontab -l | grep "$MARK"
