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
#   launch_unit.sh <unit-name> <logfile> <cmd...>
set -u
UNIT="${1:?unit}"; LOG="${2:?log}"; shift 2
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --unit="$UNIT" --collect \
  --working-directory=/home/ttuser/.coworker/wt/c12-compose-fold \
  /bin/bash -lc "$(printf '%q ' "$@") > $(printf '%q' "$LOG") 2>&1"
echo "unit=$UNIT launched at $(date -u +%FT%TZ)"
systemctl --user show "$UNIT" -p MainPID -p ActiveState
