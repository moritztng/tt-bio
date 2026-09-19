#!/usr/bin/env bash
# Stop a bracketed fold A/B session the moment it can decide, instead of when its --blocks runs out.
#
# Session 3 needs n=6 complete blocks and no more: score_bracket.py's pre-registration says the
# smallest two-sided p at n blocks is 2/2^n, so n=6 is the first n that can reach p<0.05. Blocks
# 7-11 buy resolution this row does not need and cost benchlock, which pvx-baseline and
# pvx-orchestrator have been queued behind for nearly four hours.
#
# It also bounds the two ways the session dies badly, both seen tonight:
#   STALL   a leg host-spins at 100% CPU holding a chip fd and the driver waits on it forever,
#           because apb_fold_ab.py runs subprocess.run with no timeout. Leg 512_base_4_2 did this
#           for 101 minutes on 2026-09-19 and measured nothing.
#   NOQUIET the guard correctly refuses a loud host leg after leg, so the lock is held for a
#           quiet-wait rather than for a measurement, which is the mistake pass 21 already made.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
SESSION=$WT/perf/c14_land/apb3_ab.json
DRIVER=${DRIVER:?driver pid}
WANT=${WANT:-6}
DEADLINE=${DEADLINE:?epoch}
STALL_S=${STALL_S:-2400}
LOG=$WT/perf/c14_land/session_stop_at_n.log
cd "$WT" || exit 1

complete() { python3 "$WT/perf/c14_land/complete_blocks.py" "$SESSION" 12 2>/dev/null || echo 0; }

last_change=$(date +%s)
last_n=$(complete)
echo "$(date -u +%FT%TZ) watching driver $DRIVER, have $last_n complete blocks, want $WANT" >>"$LOG"
while true; do
  sleep 60
  if ! kill -0 "$DRIVER" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) driver exited on its own" >>"$LOG"
    exit 0
  fi
  n=$(complete)
  now=$(date +%s)
  if [ "$n" != "$last_n" ]; then
    last_n=$n
    last_change=$now
    echo "$(date -u +%FT%TZ) $n complete blocks" >>"$LOG"
  fi
  reason=""
  if [ "$n" -ge "$WANT" ]; then
    reason="GOAL: $n complete blocks, n>=$WANT can reach p<0.05"
  elif [ $((now - last_change)) -ge "$STALL_S" ]; then
    reason="STALL: no new complete block in $STALL_S s"
  elif [ "$now" -ge "$DEADLINE" ]; then
    reason="DEADLINE: benchlock released for the queue"
  else
    continue
  fi
  echo "$(date -u +%FT%TZ) STOPPING -- $reason" >>"$LOG"
  # Inner first, then the driver, then its benchlock wrapper: killing the outer pid alone leaves the
  # leg holding the chip fd (fleet-kill-outer-pid-leaves-orphan-engine).
  for leg in $(pgrep -f "apb_fold_ab.py --arm" || true); do
    echo "$(date -u +%FT%TZ) kill leg $leg" >>"$LOG"
    kill -9 "$leg" 2>/dev/null
  done
  sleep 3
  kill -9 "$DRIVER" 2>/dev/null
  sleep 3
  for bl in $(pgrep -f "benchlock.sh c14-land-tail" || true); do
    echo "$(date -u +%FT%TZ) kill benchlock wrapper $bl" >>"$LOG"
    kill -9 "$bl" 2>/dev/null
  done
  echo "$(date -u +%FT%TZ) stopped; score with explode_session.py then score_bracket.py" >>"$LOG"
  exit 0
done
