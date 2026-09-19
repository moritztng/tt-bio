#!/usr/bin/env bash
# One cron tick of the measurement waiter. Installed by arm_waiter_cron.sh.
#
# WHY CRON AND NOT A DETACHED LOOP. take_when_admissible.sh is correct and its predecessor's
# defects are fixed, but it has now been destroyed twice without measuring anything, and neither
# time was its own fault:
#
#   2026-09-18 19:03Z  the K4 chain died with the host
#   2026-09-19 04:26Z  take_when_admissible.sh (pid 1479223) died with the host
#
# qb2 hard-reset at 04:26:49Z and again at 04:45:55Z, 16 minutes apart, and four times in the two
# days to 2026-09-19. `last -x` shows no matching `shutdown` record and the journal stops
# mid-line, so these are hard resets, not clean reboots. train-i-run reached the same conclusion
# independently and its crontab comment says so. A process is not the right container for a wait
# that has to outlive this box; a crontab entry is, because cron is restarted by systemd at boot.
#
# Self-limiting three ways, because an ad-hoc monitor that outlives its campaign is a known
# failure mode on this fleet: a deadline file it removes itself past, a worktree check, and an
# flock so a tick can never stack a second waiter on top of a live one.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
MARK=c14-land-tail-waiter
DEADLINE_F="$WT/perf/c14_land/waiter_deadline"
LOG="$WT/perf/c14_land/take_when_admissible.log"

disarm() {
  ( flock 9
    crontab -l 2>/dev/null | grep -v "$MARK" | crontab - 2>/dev/null
  ) 9>/tmp/c14_crontab.lock
  echo "$(date -Is) DISARMED ($1) -- crontab line removed" >>"$LOG" 2>/dev/null || true
}

[ -d "$WT" ] || { disarm "worktree gone"; exit 0; }
[ -f "$DEADLINE_F" ] || { disarm "no deadline file"; exit 0; }
DEADLINE=$(cat "$DEADLINE_F")
[ "$(date +%s)" -lt "$DEADLINE" ] || { disarm "deadline spent"; exit 0; }

# -n: if a waiter is already live, this tick is a no-op. Never two waiters on one card.
exec flock -n /tmp/c14_land_waiter.lock \
  env DEADLINE="$DEADLINE" CARD="${CARD:-1}" QUEUE="${QUEUE:-TT_BIO_APB_CONCAT_HEADS:apb2}" \
  bash "$WT/perf/c14_land/take_when_admissible.sh" >>"$LOG" 2>&1
