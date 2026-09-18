#!/usr/bin/env bash
# Launch a long job into its OWN systemd user unit, working-directory rooted in THIS slug's
# worktree. setsid does not leave the launching ssh login's cgroup scope, and session s4 of the
# C12 compose row was SIGTERM'd eight minutes in when that scope was reaped.
#   launch_unit.sh <unit-name> <logfile> <cmd...>
set -u
UNIT="${1:?unit}"; LOG="${2:?log}"; shift 2
WT=/home/ttuser/.coworker/wt/c13-land-first
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
for a in "$@"; do
  case "$a" in
    ./*|../*) echo "launch_unit.sh: '$a' is relative to YOUR cwd, and the unit runs in $WT" >&2; exit 2 ;;
  esac
done
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user --unit="$UNIT" --collect --working-directory="$WT" \
  /bin/bash -lc "$(printf '%q ' "$@") > $(printf '%q' "$LOG") 2>&1"
echo "unit=$UNIT launched at $(date -u +%FT%TZ)"
systemctl --user show "$UNIT" -p MainPID -p ActiveState
