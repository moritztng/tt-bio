#!/bin/bash
# One-shot drain harvester: poll the galaxy fleet's results.jsonl until its <RUN>_DONE
# sentinel appears, then harvest with DEST = the sshfs mount of qb1's analysis tree, so
# folds stream galaxy -> pc -> qb1 with NO pc disk landing (pc has 48 GB free; the full
# remaining harvest is ~100 GB -- a local staging tree cannot hold it, and the harvest's
# skip-if-complete makes a pruned staging re-pull forever). Fleet LAUNCHES stay pass-gated.
#
# Setsid-detached; exits after firing. Idempotent via a state file. Kill-safety: no
# pgrep patterns anywhere; managed by literal pid only.
#
# usage: watch_galaxy_drain_harvest.sh <run_id> <rung...>   e.g.: ... p27 64 256
set -u
RUN=${1:?run id, e.g. p27}; shift
RUNGS=${*:-64}
SENT=$(printf '%s' "$RUN" | tr '[:lower:]' '[:upper:]')_DONE
GB=/home/cust-team/mthuening/$RUN
STATE=$HOME/.coworker/state/deepn_harvested_$RUN
WT=$HOME/.coworker/wt/abag-xm-deepn-saturation-fullpanel
MNT=$HOME/qb1_galaxy
LOGD=$HOME/abag_xm/deepn/logs
mkdir -p "$LOGD"
LOG=$LOGD/harvest_$RUN.log

if [ -f "$STATE" ]; then echo "$(date -u) $RUN already harvested (state file); exit" >> "$LOG"; exit 0; fi
# Baseline: if the sentinel is already present at arm time the fleet drained before we
# armed -- harvest immediately rather than waiting for a second sentinel.
armed=$(date -u)
echo "$armed watcher armed for $RUN (sentinel $SENT, rungs: $RUNGS, DEST=$MNT)" >> "$LOG"
while :; do
  n=$(ssh -o BatchMode=yes -o ConnectTimeout=20 japanfold-ssh "grep -c $SENT $GB/results.jsonl 2>/dev/null" 2>/dev/null)
  case "$n" in ''|*[!0-9]*) ;; *) [ "$n" -ge 1 ] && break ;; esac
  sleep 600
done
echo "$(date -u) $SENT detected (armed $armed); harvesting into the qb1 mount" >> "$LOG"
if ! mountpoint -q "$MNT"; then
  sshfs qb1:abag_xm/deepn/galaxy "$MNT" -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    && echo "$(date -u) remounted $MNT" >> "$LOG"
fi
if ! mountpoint -q "$MNT"; then
  echo "$(date -u) MOUNT DOWN and remount failed -- NOT harvesting, no state written" >> "$LOG"
  "$HOME/.coworker/tg.sh" send "abag-xm deepn: $RUN drained but the qb1 sshfs mount on pc is down and remount failed. Harvest NOT done; re-arm the watcher after fixing the mount."
  exit 1
fi
cd "$WT" && DEST="$MNT" bash scripts/abag_xm/p25_harvest.sh "$RUN" $RUNGS >> "$LOG" 2>&1
touch "$STATE"
echo "$(date -u) harvest complete for $RUN (folds landed directly on qb1)" >> "$LOG"
"$HOME/.coworker/tg.sh" status "abag-xm deepn: $RUN drained on the galaxy; harvested straight into the qb1 tree (rungs $RUNGS). qb1 + pc labelers pick the new folds up automatically; next pass can launch the next window phase."
