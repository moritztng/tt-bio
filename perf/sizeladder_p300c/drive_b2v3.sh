#!/bin/bash
# Record one boltz-2 p300c slice on qb2. $1 = card, $2 = rungs, $3 = tries (default 12).
#
# v3 replaces v2's two broken mechanisms:
#
# 1. PID BY CWD, NOT BY pgrep. v2 read the record pid from `pgrep -f size-ladder-record`, which
#    also matches any ssh command line that merely MENTIONS the flag -- including the poll command
#    an operator runs from another host. It picked that wrapper, watched it exit seconds later,
#    declared the try dead and reaped the real record run. Same false-positive class the state doc
#    already recorded for `pgrep -f release_gate`. Ownership here is /proc/<pid>/cwd plus comm =
#    python3, which no shell and no co-tenant can match.
# 2. WEDGE BY LOG SILENCE, NOT BY CPU. v2 called a fold wedged when it burned no CPU for 150 s.
#    The boltz-2 trunk wedge on this box does the opposite: 02:40Z on 09-16 the 768 aa warm-up
#    stopped at `trunk 2/4` and sat there while its process ran 321 s of CPU in 294 s of wall,
#    i.e. above 100 %, because the host spin-waits on the device. A CPU test is structurally blind
#    to it. The live fold log's mtime is not: a healthy trunk prints a line every 8-10 s.
#    v2 also had `cur=$((cur + $(ps ...)))` break with a syntax error the moment a fold pid
#    vanished mid-loop, which is what ended the 02:37Z run after two tries instead of twelve.
#
# On a wedge: reset the chip. tt-smi 6.3.0 / tt-kmd 2.11.0 reset is per-chip, verified 09-15, so
# this cannot take a board sibling's fold down. It is still skipped if another holder's lease on
# this card is live.
CARD=$1; RUNGS=$2; TRIES=${3:-12}
WT=/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
SLUG=tt-bio-sizeladder-p300c-refresh
cd "$WT" || exit 1
HEAD=$(git rev-parse --short HEAD)
export RELEASE_GATE_FOLD_TIMEOUT=${FOLD_TIMEOUT:-420}
WEDGE_S=${WEDGE_S:-180}   # a healthy trunk prints every 8-10 s; 180 s is 20x that
WARMUP_S=${WARMUP_S:-420} # model load before the first fold line appears

mine() {
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    case "$(cat "$d/comm" 2>/dev/null)" in python3*) ;; *) continue;; esac
    case "$(readlink "$d/cwd" 2>/dev/null)" in *"$SLUG"*) echo "$p";; esac
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

lease_busy() {  # someone else holds this card right now
  f=/home/ttuser/.coworker/state/leases/tt-quietbox2-card${CARD}.json
  [ -f "$f" ] || return 1
  /home/ttuser/tt-bio-dev/env/bin/python3 - "$f" "$SLUG" <<'PY'
import json, os, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
if d.get("released") is not None or d.get("holder") == "worker:" + sys.argv[2]:
    sys.exit(1)
try:
    os.kill(int(d["pid"]), 0)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

reset_chip() {
  if lease_busy; then echo "$(date -u +%H:%M:%S) card $CARD held by another worker, skipping reset"; return; fi
  echo "$(date -u +%H:%M:%S) tt-smi -r $CARD"
  timeout 180 /home/ttuser/.local/bin/tt-smi -r "$CARD" >>perf/sizeladder_p300c/reset.log 2>&1
  echo "$(date -u +%H:%M:%S) reset exit $?"
}

live_log_age() {  # seconds since the newest boltz-2 fold log was written, -1 if none since $1
  newest=$(ls -t perf/sizegate/work/boltz2-*.log 2>/dev/null | head -1)
  [ -n "$newest" ] || { echo -1; return; }
  m=$(stat -c %Y "$newest")
  [ "$m" -lt "$1" ] && { echo -1; return; }   # older than this try: the first fold has not logged yet
  echo $(( $(date +%s) - m ))
}

for try in $(seq 1 "$TRIES"); do
  have && { echo "$(date -u +%H:%M:%S) HAVE $RUNGS at $HEAD"; exit 0; }
  reap
  LOG="perf/sizeladder_p300c/rec_b2v3_${RUNGS//,/_}_t${try}.log"
  START=$(date +%s)
  setsid nohup bash perf/sizeladder_p300c/rec_boltz2_part.sh "$CARD" "$RUNGS" >"$LOG" 2>&1 </dev/null &
  sleep 20
  PID=""
  for p in $(mine); do
    tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q -- --size-ladder-record && { PID=$p; break; }
  done
  echo "$(date -u +%H:%M:%S) try $try pid ${PID:-none} -> $LOG"
  [ -z "$PID" ] && { sleep 10; continue; }
  while kill -0 "$PID" 2>/dev/null; do
    sleep 15
    case "$(ps -o stat= -p "$PID" 2>/dev/null)" in
      T*) echo "$(date -u +%H:%M:%S) try $try STOPPED, sending CONT"
          for c in $(pgrep -P "$PID"); do kill -CONT "$c" 2>/dev/null; done
          kill -CONT "$PID" 2>/dev/null; continue ;;
    esac
    age=$(live_log_age "$START")
    if [ "$age" -lt 0 ]; then
      [ $(( $(date +%s) - START )) -ge "$WARMUP_S" ] && {
        echo "$(date -u +%H:%M:%S) try $try no fold log after ${WARMUP_S}s, relaunching"; reap; break; }
    elif [ "$age" -ge "$WEDGE_S" ]; then
      echo "$(date -u +%H:%M:%S) try $try WEDGE: fold log silent ${age}s at $(tail -1 "$(ls -t perf/sizegate/work/boltz2-*.log | head -1)" | tr -s ' ')"
      reap; reset_chip; break
    fi
  done
  if have; then echo "$(date -u +%H:%M:%S) RECORDED $RUNGS at $HEAD on try $try"; exit 0; fi
  echo "$(date -u +%H:%M:%S) try $try produced nothing"
  git checkout -- docs/size_ladder_baseline.json 2>/dev/null
done
echo "$(date -u +%H:%M:%S) GAVE UP after $TRIES tries"; exit 1
