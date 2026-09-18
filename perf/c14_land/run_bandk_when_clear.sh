#!/usr/bin/env bash
# Wait for BOTH couplings to clear (board-pair sibling + host), then take the K4 fold A/B.
# Detached on purpose: the window opens when another row finishes, not when this worker is awake.
# cwd is this rows OWN worktree, per the fleet rule about a concluded slugs worktree being torn down.
set -u
cd /home/ttuser/.coworker/wt/c14-land-tail || exit 1
G=perf/c12_orchestrator/pair_guard
CARD=3
DEADLINE=$(( $(date +%s) + 5400 ))

while :; do
  if python3 "$G/pair_idle.py" --card "$CARD" >/tmp/bandk_pair.txt 2>&1 \
     && python3 "$G/host_quiet.py" >/tmp/bandk_host.txt 2>&1; then
    echo "$(date -u +%FT%TZ) BOTH CLEAR -- taking the window"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date -u +%FT%TZ) GAVE UP: window never opened in 90 min"
    tail -3 /tmp/bandk_host.txt; tail -3 /tmp/bandk_pair.txt
    exit 75
  fi
  echo "$(date -u +%FT%TZ) waiting: $(tail -1 /tmp/bandk_pair.txt) | $(tail -1 /tmp/bandk_host.txt)"
  sleep 45
done

# Clock pinned by the caller, sampled during every fold by the harness. 1350 is the target every
# other C14 fold A/B on this part was taken at, so the number stays comparable.
python3 perf/c10_bare_baseline/force_aiclk.py 3 1350 2400 > perf/c14_land/bandk_clockpin.log 2>&1 &
PIN=$!
sleep 4

~/.coworker/scripts/benchlock.sh c14-land-tail -- \
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
  python3 perf/c14_land/apb_fold_ab.py --flag TT_BIO_SDPA_BAND_DIV_K --sizes 298 \
    --blocks 3 --folds 3 --card 3 \
    --out perf/c14_land/bandk_ab.json --cifdir perf/c14_land/bandk_ab_cifs
RC=$?
# Release the pin EXPLICITLY. Killing the holder leaves FORCE_AICLK latched on the chip for the
# next row, which is the 800 MHz latch this campaign already lost three investigations to.
kill "$PIN" 2>/dev/null
python3 perf/c10_bare_baseline/force_aiclk.py 3 0 1 >> perf/c14_land/bandk_clockpin.log 2>&1
echo "$(date -u +%FT%TZ) HARNESS EXIT $RC"
exit $RC
