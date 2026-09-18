#!/usr/bin/env bash
# The part that runs INSIDE the benchlock, for c14-p300c-256-recell.
# Separate file on purpose: nesting this in a `bash -c '...'` string cannot be checked by
# `bash -n` (it validates the quoting, not the string), and a quoting bug here would be found
# by a 16-minute unattended run instead of by a syntax check.
set -u
WT=/home/ttuser/.coworker/wt/c14-p300c-256-recell
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-1}
cd "$WT" || exit 1

echo "=== host_quiet.py inside the lock, immediately before the record ==="
"$PY" perf/c12_orchestrator/pair_guard/host_quiet.py; q=$?
"$PY" perf/c12_orchestrator/pair_guard/pair_idle.py --card "$CARD"; p=$?
echo "host_quiet=$q pair_idle=$p loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
if [ "$q" -ne 0 ] || [ "$p" -ne 0 ]; then
  echo "NOT RECORDED: the box was not quiet when the lock came free. A suspect cell is worse"
  echo "than the stale one it would replace, so nothing was measured."
  exit 3
fi

TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
TT_BIO_LEASE_HOLDER=worker:c14-p300c-256-recell \
"$PY" scripts/release_gate.py --model size-ladder --size-ladder-record \
    --size-ladder-fragment --size-ladder-models boltz2
r=$?

echo "=== after the record ==="
echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
"$PY" perf/c12_orchestrator/pair_guard/host_quiet.py; echo "host_quiet_after=$?"
exit "$r"
