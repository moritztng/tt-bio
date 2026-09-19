#!/usr/bin/env bash
# Kill a host-spin corpse holding this row's card, so a wedged fold costs minutes instead of hours.
#
# The BH p300c host-spin wedge burns a core forever while holding the chip and ignores SIGINT and
# SIGTERM, so a wedged size-ladder fold hangs the whole arm with no timeout anywhere in the gate.
# On 2026-09-19 that sat undetected for 25 minutes at openfold3 rung 768, after the warmup fold of
# the SAME rung had completed cleanly on the same chip. This watchdog changes no gate verdict: it
# turns an unbounded hang into a fast, attributable rc=137 that the next launch can re-run.
#
# It refuses to act on one reading. wedge_check.py's own docstring warns that a polling wait can
# look wedged for seconds, and a live fold is calibrated at syscw +37744 per 20 s, so requiring
# STRIKES consecutive zero-progress verdicts spanning ~5 minutes sits far outside anything live.
set -u
WT=/home/ttuser/.coworker/wt/c14-stack-land
CARD=${CARD:-1}
STRIKES=${STRIKES:-3}
PERIOD=${PERIOD:-100}
CHECK=$WT/perf/c12_orchestrator/pair_guard/wedge_check.py
OUT=/tmp/wedge_watchdog_$CARD.out
[ -f "$CHECK" ] || { echo "no wedge_check.py at $CHECK"; exit 2; }
strikes=0
while :; do
  if ! pgrep -f "release_gate.py --model size-ladder" > /dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) size-ladder gone, watchdog exiting"; exit 0
  fi
  if python3 "$CHECK" --card "$CARD" --window 20 > "$OUT" 2>&1; then
    [ $strikes -gt 0 ] && echo "$(date -u +%FT%TZ) progress resumed, strikes reset"
    strikes=0
  else
    strikes=$((strikes + 1))
    echo "$(date -u +%FT%TZ) strike $strikes/$STRIKES"
    sed -n '1,3p' "$OUT"
    if [ $strikes -ge $STRIKES ]; then
      # Kill ONLY the pids wedge_check itself named, by explicit pid, never a pattern sweep.
      for p in $(awk '/WEDGED/ {print $2}' "$OUT"); do
        argv=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
        case "$argv" in
          *multiprocessing*|*tt_bio.main*|*lever_census*)
            echo "$(date -u +%FT%TZ) SIGKILL $p :: ${argv:0:90}"; kill -9 "$p" ;;
          *)
            echo "$(date -u +%FT%TZ) REFUSING to kill $p, unexpected argv: ${argv:0:90}" ;;
        esac
      done
      strikes=0
    fi
  fi
  sleep "$PERIOD"
done
