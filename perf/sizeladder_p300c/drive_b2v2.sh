#!/bin/bash
# Record one boltz-2 p300c slice on qb2, surviving the three ways a slice dies here.
# $1 = card, $2 = rungs, $3 = tries (default 10).
#
# 1. WEDGE. A boltz-2 fold on this box hangs at the trunk with the process alive at ~0 % CPU and
#    no further log line (four sightings in pass 6 across three cards; it reads as the per-card
#    PCIe link failure QBROOT closed on). The gate's own FOLD_TIMEOUT_S is 1800 s and a timed-out
#    census fold FAILS the model rather than retrying, so at the default a single wedge costs
#    30 minutes to reach a guaranteed failure. That timeout is sized for a flaky MSA server, not
#    for a device wedge. boltz-2's 768 aa fold is ~30 s nominal and 115 s under load 7, so 600 s
#    is 5x the worst reading anyone has taken and still fails 3x faster than the default.
# 2. SIGSTOP. A setsid session leader on qb2 turns up in state T for no sender anyone has found;
#    ttx-a3 hit it the same day and wrote the fix into perf/ttx_a3/resume_after_boot.sh. A stopped
#    job is indistinguishable from a wedged one by log or CPU, and the right action is SIGCONT, not
#    a relaunch, because the arm's own process is intact. Parent AND children need the CONT.
# 3. WATCHDOG RESET. Nothing to do but retry, which is what the loop is for.
#
# OWNERSHIP IS BY /proc/<pid>/cwd, NOT BY ARGV. The gate launches each fold in its own process
# group, so killing the gate's pgid does not reach the folds; and a killed fold leaves a
# multiprocessing spawn child whose argv is only `spawn_main(...)` -- no script, no worktree, no
# model name -- which every argv or engine-name filter misses while it holds the card lease. The
# folds inherit cwd = the worktree, so cwd identifies them all and cannot hit a co-tenant.
CARD=$1; RUNGS=$2; TRIES=${3:-10}
WT=/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
cd "$WT" || exit 1
HEAD=$(git rev-parse --short HEAD)
export RELEASE_GATE_FOLD_TIMEOUT=600

mine() {  # python processes whose cwd is this worktree. Tested on comm, the actual executable,
          # NOT on argv: `pgrep -f python3` also matches any shell whose command line merely
          # MENTIONS python3, which includes the ssh wrapper that launches this driver, and reap()
          # would then kill it. Same false-positive class as the benchlock foreign-fold incident.
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    case "$(cat "$d/comm" 2>/dev/null)" in python3*) ;; *) continue;; esac
    case "$(readlink "$d/cwd" 2>/dev/null)" in
      *tt-bio-sizeladder-p300c-refresh*) echo "$p";;
    esac
  done
}
reap() { for p in $(mine); do kill -9 "$p" 2>/dev/null; done; sleep 5; }
have() {
  /home/ttuser/tt-bio-dev/env/bin/python3 - "$RUNGS" "$HEAD" <<'PY'
import json, sys
rungs, head = sys.argv[1].split(","), sys.argv[2]
try:
    e = json.load(open("docs/size_ladder_baseline.d/boltz2.json"))["cards"]["p300c"]["models"]["boltz2"]
except Exception:
    sys.exit(1)
sys.exit(0 if e.get("commit") == head and all(r in e.get("runtime_s", {}) for r in rungs) else 1)
PY
}

for try in $(seq 1 "$TRIES"); do
  have && { echo "$(date -u +%H:%M:%S) HAVE $RUNGS at $HEAD"; exit 0; }
  reap
  LOG="perf/sizeladder_p300c/rec_b2v2_${RUNGS//,/_}_t${try}.log"
  setsid nohup bash perf/sizeladder_p300c/rec_boltz2_part.sh "$CARD" "$RUNGS" >"$LOG" 2>&1 </dev/null &
  sleep 8
  PID=$(pgrep -f "size-ladder-record" | head -1)
  echo "$(date -u +%H:%M:%S) try $try pid ${PID:-none} -> $LOG"
  [ -z "$PID" ] && continue
  last=-1; stall=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 15
    case "$(ps -o stat= -p "$PID" 2>/dev/null)" in
      T*) echo "$(date -u +%H:%M:%S) try $try STOPPED, sending CONT"
          for c in $(pgrep -P "$PID"); do kill -CONT "$c" 2>/dev/null; done
          kill -CONT "$PID" 2>/dev/null; stall=0; continue ;;
    esac
    cur=0
    for p in $(mine); do cur=$((cur + $(ps -o cputimes= -p "$p" 2>/dev/null | tr -d " " || echo 0))); done
    if [ "$cur" -le "$last" ]; then stall=$((stall+15)); else stall=0; fi
    last=$cur
    if [ "$stall" -ge 150 ]; then
      echo "$(date -u +%H:%M:%S) try $try WEDGE: my folds burned no CPU for ${stall}s, relaunching"
      reap; break
    fi
  done
  if have; then echo "$(date -u +%H:%M:%S) RECORDED $RUNGS at $HEAD on try $try"; exit 0; fi
  echo "$(date -u +%H:%M:%S) try $try produced nothing"
  git checkout -- docs/size_ladder_baseline.json 2>/dev/null
done
echo "$(date -u +%H:%M:%S) GAVE UP after $TRIES tries"; exit 1
