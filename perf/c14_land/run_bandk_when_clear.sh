#!/usr/bin/env bash
# Wait for BOTH couplings to clear (board-pair sibling + host), then take the K4 fold A/B.
#
# Detached on purpose: the window opens when another row finishes, not when this worker is awake.
# cwd is this row's OWN worktree, per the fleet rule about a concluded slug's worktree being torn
# down under a live job.
#
# THE SCARCE RESOURCE IS THE WINDOW, NOT THE RUN. On 2026-09-18 this script waited 10 minutes for
# a quiet host, got about one minute of it, and spent the whole of it dying in
# ModuleNotFoundError: No module named 'torch', because it ran the harness under the system
# python3 instead of the tt-bio venv. The box was re-occupied by three rows before it could be
# retried. So: the interpreter is checked BEFORE the wait loop, not after it, and a failed run is
# retried while the window may still be open.
set -u
cd /home/ttuser/.coworker/wt/c14-land-tail || exit 1
G=perf/c12_orchestrator/pair_guard
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=3
DEADLINE=$(( $(date +%s) + 5400 ))

# Preflight the interpreter against the window, not inside it.
if ! "$PY" -c 'import torch, ttnn' 2>/dev/null; then
  echo "$(date -u +%FT%TZ) ABORT: $PY cannot import torch+ttnn. Fix this before waiting for a window."
  exit 1
fi
echo "$(date -u +%FT%TZ) interpreter ok: $PY"

run_ab() {
  "$PY" perf/c10_bare_baseline/force_aiclk.py 3 1350 2400 > perf/c14_land/bandk_clockpin.log 2>&1 &
  local pin=$!
  sleep 4
  ~/.coworker/scripts/benchlock.sh c14-land-tail -- \
    env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
    "$PY" perf/c14_land/apb_fold_ab.py --flag TT_BIO_SDPA_BAND_DIV_K --sizes 298 \
      --blocks 3 --folds 3 --card 3 \
      --out perf/c14_land/bandk_ab.json --cifdir perf/c14_land/bandk_ab_cifs
  local rc=$?
  # Release the pin EXPLICITLY. Killing the holder leaves FORCE_AICLK latched on the chip for the
  # next row, which is the 800 MHz latch this campaign already lost three investigations to.
  kill "$pin" 2>/dev/null
  "$PY" perf/c10_bare_baseline/force_aiclk.py 3 0 1 >> perf/c14_land/bandk_clockpin.log 2>&1
  return $rc
}

for attempt in 1 2; do
  while :; do
    if "$PY" "$G/pair_idle.py" --card "$CARD" >/tmp/bandk_pair.txt 2>&1 \
       && "$PY" "$G/host_quiet.py" >/tmp/bandk_host.txt 2>&1; then
      echo "$(date -u +%FT%TZ) BOTH CLEAR -- taking the window (attempt $attempt)"
      break
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      echo "$(date -u +%FT%TZ) GAVE UP: window never opened before the deadline"
      tail -3 /tmp/bandk_host.txt; tail -3 /tmp/bandk_pair.txt
      exit 75
    fi
    echo "$(date -u +%FT%TZ) waiting: $(tail -1 /tmp/bandk_pair.txt) | $(tail -1 /tmp/bandk_host.txt)"
    sleep 45
  done

  run_ab
  RC=$?
  echo "$(date -u +%FT%TZ) HARNESS EXIT $RC (attempt $attempt)"
  [ "$RC" -eq 0 ] && exit 0
  echo "$(date -u +%FT%TZ) run failed, going back to the guards"
done
exit "$RC"
