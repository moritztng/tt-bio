#!/usr/bin/env bash
# Relaunch the release-gate driver after a qb2 watchdog reset, and after any other death.
#
# Nothing on this host restarts a detached job, and qb2's mean uptime is 52 minutes over its last
# 14 boots, so without this the gate advances only when a human happens to relaunch it. Installed
# as both @reboot and */10 in ttuser's crontab: @reboot covers the resets, the ten-minute tick
# covers a driver that died for any other reason.
#
# Self-limiting in three ways, because a cron entry outlives the task that wrote it: it does
# nothing once the driver has logged GATE_DRIVER_DONE, nothing if a driver is already running, and
# nothing if the worktree has been torn down.
set -u
# SINGLE INSTANCE, and the lock is taken BEFORE anything that sleeps.
#
# The first version guarded with `pgrep` for a live driver. That is not enough: after the
# 2026-09-15T09:38:28Z boot the @reboot entry and the */10 tick fired together, BOTH found no driver
# because neither had started one yet, both slept until uptime 150s, and both then launched. Two
# drivers ran the same arm on the same card in the same worktree, their two census runs collided in
# the shared work dir, and the arm reported
#   FAIL rung 256 warm-up: census fold exited 1: FileNotFoundError ... /dumps/pid10375.json
# which reads exactly like a gate failure and was not one. A check-then-act guard cannot fix a race
# it sits inside; the lock can, because it is held across the sleep and the launch.
exec 9>/tmp/ttx-a3-sdpa-ship-remerge.resume.lock || exit 0
flock -n 9 || exit 0
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge
[ -d "$WT" ] || exit 0
cd "$WT" || exit 0
PROG=perf/ttx_a3/gate2/progress
# While PAUSE exists the main gate stands down and the rf3/1088 attribution control owns the card
# instead. That control is the one measurement that can turn this into a NO-GO, so it gets the box
# first, and it needs the same boot survival as the gate: qb2 reset 28 minutes into the boot the
# stall was found on. Remove PAUSE to hand the card back to the gate.
if [ -f perf/ttx_a3/gate2/PAUSE ]; then
  grep -q RF3_1088_DONE perf/ttx_a3/gate2/rf3_1088/progress 2>/dev/null && exit 0
  pgrep -f "^bash perf/ttx_a3/attr_rf3_1088\.sh$" > /dev/null && exit 0
  up=$(cut -d. -f1 /proc/uptime)
  [ "$up" -lt 150 ] && sleep $((150 - up))
  mkdir -p perf/ttx_a3/gate2/rf3_1088
  printf '%s resume_after_boot relaunching ATTR (uptime %ss)\n' \
         "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(cut -d. -f1 /proc/uptime)" \
         >> perf/ttx_a3/gate2/rf3_1088/progress
  setsid nohup bash perf/ttx_a3/attr_rf3_1088.sh \
    >> perf/ttx_a3/gate2/rf3_1088/driver.log 2>&1 < /dev/null &
  exit 0
fi
grep -q GATE_DRIVER_DONE "$PROG" 2>/dev/null && exit 0
# ANCHORED argv match. A cheap early-out only; the flock above is what makes this safe. An unanchored `pgrep -f "bash perf/ttx_a3/gate_drive.sh"` also matches any
# shell whose command line merely CONTAINS that text, which includes the `bash -c` wrapper an ssh
# invocation of this very script runs under: the guard then reports a live driver, this exits, and
# nothing relaunches. The driver's own command line is exactly `bash perf/ttx_a3/gate_drive.sh`,
# and a wrapper's is `bash -c ...`, so anchoring separates them.
pgrep -f "^bash perf/ttx_a3/gate_drive\.sh$" > /dev/null && exit 0

# Let the box finish coming up before opening a card: the driver's first act is a device open and
# the ARC is not ready the instant systemd starts running cron jobs.
up=$(cut -d. -f1 /proc/uptime)
[ "$up" -lt 150 ] && sleep $((150 - up))

printf '%s resume_after_boot relaunching (uptime %ss)\n' \
       "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(cut -d. -f1 /proc/uptime)" >> "$PROG"
setsid nohup env GATE_WT="$WT" GATE_OUT=perf/ttx_a3/gate2 GATE_CARD=0 \
  GATE_HOLDER=worker:ttx-a3-sdpa-ship-remerge \
  bash perf/ttx_a3/gate_drive.sh >> perf/ttx_a3/gate2/driver.log 2>&1 < /dev/null &
exit 0
