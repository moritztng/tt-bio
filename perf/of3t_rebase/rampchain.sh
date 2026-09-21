#!/usr/bin/env bash
# Wait for the float64 FD job to finish, then capture blocks 42..46 and run the arms.
#
# Serialised deliberately: the FD validation and this capture are both float64 CPU jobs on the
# same box, and running float64 jobs concurrently here has cost about a third of the speed of
# each before. FAILURE STOPS THE CHAIN -- a capture that dies must not hand a missing boundary
# to the arm step, which would then "skip" it and report a short table as if it were complete.
set -uo pipefail
L=/home/ttuser/of3t_rebase/run/rampchain.log
S=/home/ttuser/of3t_rebase/run/rampchain.status
RUNID=$(date -u +%Y%m%dT%H%M%SZ)
exec >>"$L" 2>&1
# The log is APPEND-mode so history survives, which means a bare grep for a terminal marker can
# match a PREVIOUS run's. It already did once: a waiter armed on "RAMPCHAIN_ABORT" fired
# instantly on the stale marker left by the aborted first launch, and reported a running capture
# as finished. Every terminal marker therefore carries $RUNID, and $S is TRUNCATED at the start
# of each run and holds exactly one line -- so `cat $S` is unambiguous without any log parsing.
: >"$S"
echo "=== rampchain start $RUNID ==="
while pgrep -f "bundle_min.py .*out_043_fd" >/dev/null; do sleep 60; done
echo "=== fd job clear $(date -u +%FT%TZ) ==="
bash /home/ttuser/of3t_rebase/wt/perf/of3t_rebase/capramp.sh
rc=$?
echo "=== capramp rc=$rc $(date -u +%FT%TZ) ==="
if [ "$rc" -ne 0 ]; then echo "RAMPCHAIN_ABORT $RUNID capture failed rc=$rc" | tee "$S"; exit "$rc"; fi
for b in 42 43 44 45 46; do
  [ -f /home/ttuser/of3t_rebase/cap043_ramp/block${b}_boundary.pt ] || {
    echo "RAMPCHAIN_ABORT $RUNID missing capture for block $b -- refusing a short arm table" | tee "$S"; exit 3; }
done
CARD=0 bash /home/ttuser/of3t_rebase/wt/perf/of3t_rebase/ramparms.sh
rc=$?
echo "=== ramparms rc=$rc $(date -u +%FT%TZ) ==="
[ "$rc" -eq 0 ] && echo "RAMPCHAIN_ALLDONE $RUNID $(date -u +%FT%TZ)" | tee "$S" \
  || echo "RAMPCHAIN_ABORT $RUNID arms failed rc=$rc" | tee "$S"
exit "$rc"
