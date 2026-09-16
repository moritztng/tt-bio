#!/usr/bin/env bash
# Kill a fold that wedged, so a ladder lane heals itself instead of sitting out its 3600 s timeout.
#
# boltz-2 wedges at `trunk 0/4` on this box: the fold stops writing, holds the card and burns no
# CPU. It cost three retries in one pass, each needing a human to SIGINT the device child. A fold
# writes its log within ~3 min of the previous line at every p300c rung, so STALE=480 s of silence
# is a wedge and not a slow rung.
#
# Identification is by LABEL, not by pattern: every census wrapper carries `--label <rung>` and
# writes work/<label>.log, so the stale log names the exact process to kill and a sibling lane's
# fold on another card is never touched. SIGINT first, because the device close is what releases
# the card; SIGKILL leaves the chip dirty for the next arm.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2
WORK="$WT/perf/sizegate/work"
PROG="$WT/perf/ttx_a3/gate6/progress"
STALE=${STALE:-480}
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }
while :; do
  sleep 60
  for wrapper in $(pgrep -f 'lever_census.py --tt-bio' || true); do
    label=$(tr '\0' '\n' < "/proc/$wrapper/cmdline" 2>/dev/null | grep -A1 '^--label$' | tail -1)
    [ -n "$label" ] || continue
    f="$WORK/$label.log"
    [ -f "$f" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    [ "$age" -lt "$STALE" ] && continue
    grep -q 'trunk\|prep\|diffusion' "$f" 2>/dev/null || continue
    dump="$WT/perf/sizegate/hang/${label}-wedge-$(date -u +%H%M%S).txt"
    mkdir -p "$(dirname "$dump")"
    { date -u; echo "stale ${age}s: $f"; tail -3 "$f"
      for p in $(pgrep -P "$wrapper"; pgrep -f 'multiprocessing.spawn'); do
        echo "--- pid $p"; ~/.local/bin/py-spy dump --pid "$p" 2>&1 | head -30
      done; } > "$dump" 2>&1
    kids=$(pgrep -P "$wrapper" | tr '\n' ' ')
    grand=""
    for k in $kids; do grand="$grand $(pgrep -P "$k" | tr '\n' ' ')"; done
    log "WEDGE-KILL $label stale ${age}s, SIGINT$grand$kids (dump $(basename "$dump"))"
    # shellcheck disable=SC2086
    kill -INT $grand $kids 2>/dev/null
  done
done
