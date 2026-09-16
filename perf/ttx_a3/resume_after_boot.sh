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
# The lock is keyed on the WORKTREE as well as the run dir. Two worktrees re-gating the same
# lever would otherwise share one lock file, and the second one's tick would exit silently
# forever at the flock below while looking exactly like "a driver is already running".
exec 9>"/tmp/$(basename "${GATE_WT:-ttx-a3-sdpa-ship-remerge}").resume.${GATE_OUT##*/}.lock" || exit 0
flock -n 9 || exit 0
# Run dir, card and worker host are inputs, set by the crontab line that installs this. They were
# constants naming gate3/card0/qb2, and when the re-gate moved to qb1 card 3 the relaunch would
# have quietly resumed a DIFFERENT run on a DIFFERENT card than the one this pass started -- the
# progress file it guards on and the progress file the driver writes would not have been the same
# file, so the DONE check never fires and the gate restarts forever.
# WT is an input for the same reason CARD/RUN/WORKER are: this gate has now run from three
# worktrees, and a hardcoded path resumes another worker's tree on this worker's card grant.
WT="${GATE_WT:-/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge}"
RUN="${GATE_OUT:-perf/ttx_a3/gate3}"
CARD="${GATE_CARD:-0}"
WORKER="${GATE_WORKER:-tt-quietbox2}"
# What to relaunch is an input too. The gate now runs as one serial chain (serial_gate.sh) so
# that two device lanes stop resetting this box, and the guard below has to match whatever it
# actually launched -- a guard anchored on the driver argv stands down for a chain that never
# started, and then nothing relaunches.
CMD="${GATE_CMD:-perf/ttx_a3/gate_drive.sh}"
DONE_MARK="${GATE_DONE_MARK:-GATE_DRIVER_DONE}"
[ -d "$WT" ] || exit 0
cd "$WT" || exit 0
PROG=$RUN/progress
# The two redirects below (PROG and driver.log) both fail if the run dir is absent, and a failed
# redirect means the relaunch never happens -- silently, because cron discards the error. The
# driver mkdir -p's its own OUT, but it cannot do so before the line that launches it.
mkdir -p "$RUN" || exit 0
# PAUSE means stand down completely. It used to relaunch whichever attribution control owned the
# card, naming that control by path -- which broke the moment those one-off scripts were unified
# into attr_perf_model.sh and deleted, leaving this branch pointing at a file that no longer exists.
# Attribution runs are launched by hand and are short; the gate is the only thing worth restarting
# unattended.
# A PAUSE holding a unix epoch is a TIMED hold: stand down until then, then clear it here and
# launch. An empty PAUSE is an indefinite hold, as before. The timed form exists because the
# thing this gate waits on is usually a sibling campaign's exclusive window with a known end
# (a soak's own `--until`), and a worker turn is far shorter than that window -- without it the
# hold has to be lifted by hand, which means the box sits idle until someone happens to look.
if [ -f "$RUN/PAUSE" ]; then
  _until=$(tr -dc 0-9 < "$RUN/PAUSE" 2>/dev/null | head -c 20)
  if [ -n "$_until" ] && [ "$(date +%s)" -ge "$_until" ]; then
    printf '%s resume_after_boot: timed PAUSE (until %s) expired, clearing and launching\n' \
           "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$_until" >> "$PROG"
    rm -f "$RUN/PAUSE"
  else
    exit 0
  fi
fi
grep -q "$DONE_MARK" "$PROG" 2>/dev/null && exit 0
# ANCHORED argv match. A cheap early-out only; the flock above is what makes this safe. An unanchored `pgrep -f "bash perf/ttx_a3/gate_drive.sh"` also matches any
# shell whose command line merely CONTAINS that text, which includes the `bash -c` wrapper an ssh
# invocation of this very script runs under: the guard then reports a live driver, this exits, and
# nothing relaunches. The driver's own command line is exactly `bash perf/ttx_a3/gate_drive.sh`,
# and a wrapper's is `bash -c ...`, so anchoring separates them.
# A LIVE driver means running, not merely existing. On 2026-09-15 qb2's driver and its `timeout`
# child were both found in state T, SIGSTOPped: the arm's 500 s timer could not fire, so no rc was
# ever recorded, and this guard saw a process and stood down every ten minutes. The gate sat dead
# for 11 minutes while looking perfectly healthy to every check that asked "is a driver running?".
# A stopped driver is the one case where the answer to that question is yes and the right action is
# still to intervene -- and the intervention is SIGCONT, not a relaunch, because the arm's own
# process is still there and killing it throws the arm away. One CONT restored it and the arm
# recorded rc=124 within seconds.
# Scoped to THIS worktree by cwd. The argv is identical from every worktree, so an unscoped
# match lets a sibling worktree's driver satisfy this guard and this run then never starts --
# the same worktree-wide-reap defect that killed a sibling campaign's folds on 2026-09-16.
dpid=""
for _p in $(pgrep -f "^bash ${CMD//./\\.}$"); do
  [ "$(readlink -f "/proc/$_p/cwd" 2>/dev/null)" = "$(readlink -f "$WT")" ] || continue
  dpid="$_p"; break
done
if [ -n "$dpid" ]; then
  case "$(ps -o stat= -p "$dpid" 2>/dev/null)" in
    T*) printf '%s resume_after_boot: driver %s was STOPPED, sending CONT\n' \
               "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$dpid" >> "$PROG"
        # The child holds the arm's timer, so it has to come back too, or the arm still never ends.
        for c in $(pgrep -P "$dpid"); do kill -CONT "$c" 2>/dev/null; done
        kill -CONT "$dpid" 2>/dev/null ;;
  esac
  exit 0
fi

# Let the box finish coming up before opening a card: the driver's first act is a device open and
# the ARC is not ready the instant systemd starts running cron jobs.
up=$(cut -d. -f1 /proc/uptime)
[ "$up" -lt 150 ] && sleep $((150 - up))

printf '%s resume_after_boot relaunching (uptime %ss)\n' \
       "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(cut -d. -f1 /proc/uptime)" >> "$PROG"
# 9>&- CLOSES THE LOCK FD IN THE CHILD, and it is not a tidiness detail -- without it this whole
# guard inverts. fd 9 carries the flock, the driver inherits every open fd, and the driver outlives
# this script by hours, so the lock stays held for the driver's entire life. Every */10 tick then
# dies at `flock -n 9 || exit 0` and the watchdog can never intervene AT THE ONE TIME IT IS NEEDED:
# a driver that exists but is stopped. That is not hypothetical -- gate5's driver sat in state T for
# seven minutes with its finished arm's `timeout` child a zombie beside it, the arm's rc never
# recorded, and `lsof` showed pid 3947 holding this very lock. The flock must cover this script's
# own check-and-launch and nothing beyond it.
setsid nohup env GATE_WT="$WT" GATE_OUT="$RUN" GATE_CARD="$CARD" GATE_WORKER="$WORKER" \
  GATE_SKIP_NEUT="${GATE_SKIP_NEUT:-0}" \
  GATE_HOLDER="${GATE_HOLDER:-worker:ttx-a3-sdpa-ship-remerge}" \
  bash "$CMD" >> "$RUN/driver.log" 2>&1 < /dev/null 9>&- &
exit 0
