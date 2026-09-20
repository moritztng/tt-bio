#!/usr/bin/env bash
# Wait for the float64 FD job to finish, then capture blocks 42..46 and run the arms.
#
# Serialised deliberately: the FD validation and this capture are both float64 CPU jobs on the
# same box, and running float64 jobs concurrently here has cost about a third of the speed of
# each before. FAILURE STOPS THE CHAIN -- a capture that dies must not hand a missing boundary
# to the arm step, which would then "skip" it and report a short table as if it were complete.
set -uo pipefail
L=/home/ttuser/of3t_rebase/run/rampchain.log
exec >>"$L" 2>&1
echo "=== rampchain start $(date -u +%FT%TZ) ==="
while pgrep -f "bundle_min.py .*out_043_fd" >/dev/null; do sleep 60; done
echo "=== fd job clear $(date -u +%FT%TZ) ==="
bash /home/ttuser/of3t_rebase/wt/perf/of3t_rebase/capramp.sh
rc=$?
echo "=== capramp rc=$rc $(date -u +%FT%TZ) ==="
if [ "$rc" -ne 0 ]; then echo "RAMPCHAIN_ABORT capture failed rc=$rc"; exit "$rc"; fi
for b in 42 43 44 45 46; do
  [ -f /home/ttuser/of3t_rebase/cap043_ramp/block${b}_boundary.pt ] || {
    echo "RAMPCHAIN_ABORT missing capture for block $b -- refusing to run a short arm table"; exit 3; }
done
CARD=0 bash /home/ttuser/of3t_rebase/wt/perf/of3t_rebase/ramparms.sh
rc=$?
echo "=== ramparms rc=$rc $(date -u +%FT%TZ) ==="
[ "$rc" -eq 0 ] && echo "RAMPCHAIN_ALLDONE $(date -u +%FT%TZ)" || echo "RAMPCHAIN_ABORT arms failed rc=$rc"
exit "$rc"
