#!/bin/bash
# p29 REBUILD-drain harvester (closes the pass-405 one-shot gap): the first
# watch_galaxy_drain_harvest.sh sentinel spends itself on the first P29_DONE and its
# deepn_harvested_p29 state file then suppresses any plain re-arm -- and the stale
# first P29_DONE line would fire a naive watcher instantly. This watcher baselines
# the P29_DONE count at arm time and fires only when the count INCREASES (the
# relaunched fleet appends a second P29_DONE when the rebuild set drains).
#
# Re-armable across multiple rebuild rounds: the state file records the sentinel
# count it harvested at; a later arm with a higher current count re-arms instead
# of exiting. Otherwise identical to the first sentinel: harvest straight into the
# qb1 sshfs mount (no pc landing), propagate linked-chunk labels, tg status.
#
# Setsid-detached; exits after firing. Kill-safety: kills nothing, no pgrep
# patterns; managed by literal pid only. Arm AFTER p29_rebuild_claims.sh + fleet
# relaunch (any time before the rebuild drain completes).
set -u
RUN=p29
RUNG=256
SENT=P29_DONE
GB=/home/cust-team/mthuening/$RUN
STATE=$HOME/.coworker/state/deepn_harvested_${RUN}_rebuild
WT=$HOME/.coworker/wt/abag-xm-deepn-saturation-fullpanel
MNT=$HOME/qb1_galaxy
LOGD=$HOME/abag_xm/deepn/logs
mkdir -p "$LOGD"
LOG=$LOGD/harvest_${RUN}_rebuild.log

sentinel_count() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 japanfold-ssh "grep -c $SENT $GB/results.jsonl 2>/dev/null" 2>/dev/null
}

n0=$(sentinel_count)
case "$n0" in ''|*[!0-9]*)
  echo "$(date -u) cannot read baseline $SENT count (ssh?); NOT armed" >> "$LOG"; exit 1 ;;
esac
if [ -f "$STATE" ]; then
  done_at=$(cat "$STATE")
  if [ -n "$done_at" ] && [ "$done_at" -ge "$n0" ]; then
    echo "$(date -u) rebuild drain at count=$n0 already harvested (state=$done_at); exit" >> "$LOG"
    exit 0
  fi
fi
echo "$(date -u) rebuild watcher armed (baseline $SENT count=$n0, rung $RUNG, DEST=$MNT)" >> "$LOG"

while :; do
  n=$(sentinel_count)
  case "$n" in ''|*[!0-9]*) ;; *) [ "$n" -gt "$n0" ] && break ;; esac
  sleep 600
done
echo "$(date -u) $SENT count $n0 -> $n (rebuild drained); harvesting into the qb1 mount" >> "$LOG"

if ! mountpoint -q "$MNT"; then
  sshfs qb1:abag_xm/deepn/galaxy "$MNT" -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    && echo "$(date -u) remounted $MNT" >> "$LOG"
fi
if ! mountpoint -q "$MNT"; then
  echo "$(date -u) MOUNT DOWN and remount failed -- NOT harvesting, no state written" >> "$LOG"
  "$HOME/.coworker/tg.sh" send "abag-xm deepn: p29 rebuild drained but the qb1 sshfs mount on pc is down and remount failed. Harvest NOT done; re-arm p29_rebuild_harvest_watch.sh after fixing the mount."
  exit 1
fi
cd "$WT" && DEST="$MNT" bash scripts/abag_xm/p25_harvest.sh "$RUN" "$RUNG" >> "$LOG" 2>&1
scp -q "$WT/scripts/abag_xm/propagate_linked_labels.py" qb1:/tmp/ >> "$LOG" 2>&1
ssh qb1 'nice -15 python3 /tmp/propagate_linked_labels.py propagate' >> "$LOG" 2>&1
echo "$n" > "$STATE"
echo "$(date -u) rebuild harvest+propagate complete for $RUN (count=$n)" >> "$LOG"
"$HOME/.coworker/tg.sh" status "abag-xm deepn: p29 REBUILD set drained on the galaxy; harvested into the qb1 tree (rung $RUNG), linked-chunk labels propagated. Labelers pick the folds up automatically; next pass can run --deep + datasheet."
