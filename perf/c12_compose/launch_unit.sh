#!/usr/bin/env bash
# Launch a long job into its OWN systemd user unit instead of `setsid nohup`.
#
# Why: session s4 was SIGTERM'd eight minutes in when the cgroup scope of the ssh login that
# launched it (session-822.scope) was reaped -- journal, to the second, with s4's own 8min15s CPU
# and 1.9G peak against it. setsid does not leave that scope. KillUserProcesses is already at its
# default `no` and Linger=yes for this user, so neither of the two settings usually reached for is
# the problem. A transient user unit is owned by the user manager, which lingers, so no ssh login
# coming or going can take it down.
#
# The working directory is the REPO ROOT, not the caller's cwd, so every path in the command
# must be absolute. A relative ./run.sh exited 127 before it opened anything (session s6,
# 23:32:35Z) and cost the only clean board pair of that pass: a sibling row took the card in
# the two minutes it took to notice. So the wrapper checks it rather than documenting it.
#
#   launch_unit.sh <unit-name> <logfile> <cmd...>
set -u
UNIT="${1:?unit}"; LOG="${2:?log}"; shift 2
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
for a in "$@"; do
  case "$a" in
    ./*|../*) echo "launch_unit.sh: '$a' is relative to YOUR cwd, and the unit runs in the repo" >&2
              echo "launch_unit.sh: root. Use an absolute path." >&2; exit 2 ;;
  esac
done
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --unit="$UNIT" --collect \
  --working-directory=/home/ttuser/.coworker/wt/c12-compose-fold \
  /bin/bash -lc "$(printf '%q ' "$@") > $(printf '%q' "$LOG") 2>&1"
echo "unit=$UNIT launched at $(date -u +%FT%TZ)"
systemctl --user show "$UNIT" -p MainPID -p ActiveState
