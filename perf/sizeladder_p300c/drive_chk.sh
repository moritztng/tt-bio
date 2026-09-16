#!/bin/bash
# Run one size-ladder CHECK slice on qb2 until it produces a verdict. $1 = model, $2 = card,
# $3 = rungs, $4 = tries (default 8).
#
# Same two mechanisms as drive_b2v3.sh, for the same two reasons: own the pid by /proc/<pid>/cwd
# rather than by `pgrep -f`, which also matches any ssh command line that merely mentions the
# flag; and call a fold wedged by LOG SILENCE, not by CPU, because the boltz-2 trunk wedge on
# this box spin-waits above 100 % CPU while printing nothing.
#
# The done test is the verdict line, not the baseline file: a check writes no baseline.
MODEL=$1; CARD=$2; RUNGS=$3; TRIES=${4:-8}
WT=/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
SLUG=tt-bio-sizeladder-p300c-refresh
cd "$WT" || exit 1
case "$MODEL" in
  boltz2)   SCRIPT=perf/sizeladder_p300c/chk_boltz2_part.sh ;;
  esmfold2) SCRIPT=perf/sizeladder_p300c/chk_esm_part.sh ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
export RELEASE_GATE_FOLD_TIMEOUT=${FOLD_TIMEOUT:-420}
WEDGE_S=${WEDGE_S:-180}
WARMUP_S=${WARMUP_S:-420}

mine() {
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    case "$(cat "$d/comm" 2>/dev/null)" in python3*) ;; *) continue;; esac
    case "$(readlink "$d/cwd" 2>/dev/null)" in *"$SLUG"*) ;; *) continue;; esac
    case "$(tr '\0' '\n' <"$d/environ" 2>/dev/null | grep '^TT_VISIBLE_DEVICES=')" in
      "TT_VISIBLE_DEVICES=$CARD") echo "$p";;
    esac
  done
}
reap() { for p in $(mine); do kill -9 "$p" 2>/dev/null; done; sleep 5; }

lease_busy() {
  f=/home/ttuser/.coworker/state/leases/tt-quietbox2-card${CARD}.json
  [ -f "$f" ] || return 1
  /home/ttuser/tt-bio-dev/env/bin/python3 - "$f" "$SLUG" <<"PY"
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

live_log_age() {
  newest=$(ls -t perf/sizegate/work/${MODEL}-*.log 2>/dev/null | head -1)
  [ -n "$newest" ] || { echo -1; return; }
  m=$(stat -c %Y "$newest")
  [ "$m" -lt "$1" ] && { echo -1; return; }
  echo $(( $(date +%s) - m ))
}

for try in $(seq 1 "$TRIES"); do
  reap
  LOG="perf/sizeladder_p300c/chkd_${MODEL}_${RUNGS//,/_}_t${try}.log"
  START=$(date +%s)
  setsid nohup bash "$SCRIPT" "$CARD" "$RUNGS" >"$LOG" 2>&1 </dev/null &
  sleep 20
  PID=""
  for p in $(mine); do
    tr "\0" " " </proc/$p/cmdline 2>/dev/null | grep -q -- "--size-ladder-models $MODEL" && { PID=$p; break; }
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
      echo "$(date -u +%H:%M:%S) try $try WEDGE: fold log silent ${age}s at $(tail -1 "$(ls -t perf/sizegate/work/${MODEL}-*.log | head -1)" | tr -s " ")"
      reap; reset_chip; break
    fi
  done
  if grep -qE "^GATE (PASS|FAIL)" "$LOG"; then
    echo "$(date -u +%H:%M:%S) VERDICT on try $try: $(grep -E "^GATE (PASS|FAIL)" "$LOG" | head -1)"
    exit 0
  fi
  echo "$(date -u +%H:%M:%S) try $try produced no verdict"
done
echo "$(date -u +%H:%M:%S) GAVE UP after $TRIES tries"; exit 1
