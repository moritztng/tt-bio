#!/bin/bash
# Restart the three chains if they are not running. Cron runs this every 5 minutes and at boot.
#
# WHY: qb2 went down twice on 2026-09-21 while this row ran, 12:45:32Z and 13:27:09Z, 42 minutes
# apart, each after ~60 s of `tenstorrent 0000:04:00.0: Failed to set initial power state: -5`.
# A 20-rung arm is 61 min on the device and 2.35 h on the reference, so a host with a ~40 min
# mean time between reboots only ever finishes an arm if something restarts it unattended and
# the arm picks up where it stopped. `resume.py` does the second half; this does the first.
#
# Self-limiting three ways, because a cron entry outlives the worktree it was written for:
#   - the worktree is gone   -> remove this crontab entry and exit
#   - $R/STOP exists         -> exit
#   - every arm has rc=0     -> remove this crontab entry and exit
set -u
WT=/home/ttuser/.coworker/wt/of3t-trajwide
R=/home/ttuser/of3t_runs/trajwide
L=$R/supervise.log
say() { echo "$(date -u +%FT%TZ) supervise: $*" >> "$L"; }
unhook() { crontab -l 2>/dev/null | grep -v "of3t_trajwide/supervise.sh" | crontab -; }

[ -d "$WT" ] || { unhook; exit 0; }
mkdir -p "$R"
[ -e "$R/STOP" ] && exit 0

ALL_DONE=1
for a in theirs theirs_aa2 shipped shipped_aa2 permute stale norebind zero; do
  grep -q "rc=0" "$R/$a.done" 2>/dev/null || ALL_DONE=0
done
if [ "$ALL_DONE" = 1 ]; then say "every arm has an rc=0 marker, unhooking"; unhook; exit 0; fi

# pgrep over the full command line would match this script, which names every chain below, so
# each pattern is anchored on the chain script's own argv and this pid is excluded.
running() { pgrep -u ttuser -f "$1" | grep -qv "^$$\$"; }

# An orphaned python arm outlives its chain when the chain is killed but the host is not. It
# still holds a card and its flock died with the chain, so starting a card chain now would open
# the same device twice. Wait a cycle instead.
if pgrep -u ttuser -f "trajwide.py --side ours" >/dev/null 2>&1 && ! running "of3t_trajwide/run_arms.sh"; then
  say "an ours arm is running with no chain over it; leaving the card chains alone this cycle"
  ORPHAN=1
else
  ORPHAN=0
fi

start() {                       # start <tag> <pattern> <cmd...>
  local tag=$1 pat=$2
  shift 2
  running "$pat" && return
  say "starting $tag"
  cd "$WT" || return
  setsid nohup "$@" >> "$R/chain_${tag}.out" 2>&1 &
}

# The recorder is the only thing that puts a rung on the branch, and the branch is the only
# storage on this fleet that survives a qb2 reboot. It has to come back too.
start recorder "of3t_trajwide/recorder.py" /home/ttuser/tt-bio-dev/env/bin/python "$WT/perf/of3t_trajwide/recorder.py" 180

start theirs "of3t_trajwide/run_theirs.sh" "$WT/perf/of3t_trajwide/run_theirs.sh"
if [ "$ORPHAN" = 0 ]; then
  start card1 "of3t_trajwide/run_arms.sh 1 " "$WT/perf/of3t_trajwide/run_arms.sh" 1 shipped shipped_aa2 permute stale norebind zero
  start card0 "of3t_trajwide/run_arms.sh 0 " "$WT/perf/of3t_trajwide/run_arms.sh" 0 zero norebind stale permute shipped_aa2 shipped
fi
