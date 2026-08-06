#!/bin/bash
# Recurring p29 incremental harvester (closes the pass-329 gap): the one-shot
# sweep-2 relauncher fires only ONCE, leaving a ~10h hole between sweep-2's exit
# and the drain sentinel's final harvest. This loop re-launches the identical
# skip-if-complete sweep after each exit until P29_DONE appears in the fleet's
# results.jsonl -- then it STOPS launching and exits, because the armed drain
# sentinel (watch_galaxy_drain_harvest.sh) owns the final harvest and the
# deepn_harvested_p29 state file.
#
# Guards inherited from the pass-329 watcher + pass-303 one-survivor rule:
# never runs two p29 harvests concurrently (the sentinel's final harvest also
# matches the pgrep pattern, so this loop never races it from this side), state
# file defers to the sentinel, mount verified/remounted before every sweep.
# Kill-safety: this script kills nothing; it only launches and waits.
# Setsid-detached; 16h hard cap.
set -u
RUN=p29
RUNG=256
SENT=P29_DONE
GB=/home/cust-team/mthuening/$RUN
STATE=$HOME/.coworker/state/deepn_harvested_$RUN
WT=$HOME/.coworker/wt/abag-xm-deepn-saturation-fullpanel
MNT=$HOME/qb1_galaxy
LOG=/tmp/p29_harvest_sweep_loop.log
INTER_SWEEP_S=600
end=$(( $(date +%s) + 57600 ))

sentinel_seen() {  # fleet drained?
  local n
  n=$(ssh -o BatchMode=yes -o ConnectTimeout=20 japanfold-ssh "grep -c $SENT $GB/results.jsonl 2>/dev/null" 2>/dev/null)
  case "$n" in ''|*[!0-9]*) return 1 ;; *) [ "$n" -ge 1 ] ;; esac
}

echo "$(date -u +%FT%TZ) sweep loop armed (until $SENT or 16h cap)" >> "$LOG"
while :; do
  [ -f "$STATE" ] && { echo "$(date -u +%FT%TZ) state file present (sentinel fired); exit" >> "$LOG"; exit 0; }
  sentinel_seen && { echo "$(date -u +%FT%TZ) $SENT present; drain sentinel owns the final harvest; exit" >> "$LOG"; exit 0; }
  [ "$(date +%s)" -ge "$end" ] && { echo "$(date -u +%FT%TZ) 16h cap; exit" >> "$LOG"; exit 0; }

  # one-survivor: wait out any running p29 harvest (ours or the sentinel's)
  while pgrep -f "p25_harvest.s[h] $RUN" >/dev/null; do
    sleep 60
    [ -f "$STATE" ] && exit 0
    [ "$(date +%s)" -ge "$end" ] && exit 0
  done
  sentinel_seen && continue

  if ! mountpoint -q "$MNT"; then
    sshfs qb1:abag_xm/deepn/galaxy "$MNT" -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
      && echo "$(date -u +%FT%TZ) remounted $MNT" >> "$LOG"
  fi
  if ! mountpoint -q "$MNT"; then
    echo "$(date -u +%FT%TZ) MOUNT DOWN and remount failed; retry next cycle" >> "$LOG"
    sleep "$INTER_SWEEP_S"
    continue
  fi

  echo "$(date -u +%FT%TZ) launching sweep" >> "$LOG"
  ( cd "$WT" && DEST="$MNT" bash scripts/abag_xm/p25_harvest.sh "$RUN" "$RUNG" ) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) sweep exited rc=$?" >> "$LOG"
  sleep "$INTER_SWEEP_S"
done
