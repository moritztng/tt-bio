#!/bin/bash
# One-shot drain harvester: poll the galaxy fleet's results.jsonl until its <RUN>_DONE
# sentinel appears, then harvest on pc (qb1 cannot reach the galaxy) and rsync-relay to
# qb1 so the labelers start without waiting for a pass. Fleet LAUNCHES stay pass-gated
# (determinism checks); this automates only the read-only harvest + relay + notify.
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
LOGD=$HOME/abag_xm/deepn/logs
mkdir -p "$LOGD"
LOG=$LOGD/harvest_$RUN.log

if [ -f "$STATE" ]; then echo "$(date -u) $RUN already harvested (state file); exit" >> "$LOG"; exit 0; fi
# Baseline: if the sentinel is already present at arm time the fleet drained before we
# armed -- harvest immediately rather than waiting for a second sentinel.
armed=$(date -u)
echo "$armed watcher armed for $RUN (sentinel $SENT, rungs: $RUNGS)" >> "$LOG"
while :; do
  n=$(ssh -o BatchMode=yes -o ConnectTimeout=20 japanfold-ssh "grep -c $SENT $GB/results.jsonl 2>/dev/null" 2>/dev/null)
  case "$n" in ''|*[!0-9]*) ;; *) [ "$n" -ge 1 ] && break ;; esac
  sleep 600
done
echo "$(date -u) $SENT detected (armed $armed); harvesting" >> "$LOG"
cd "$WT" && bash scripts/abag_xm/p25_harvest.sh "$RUN" $RUNGS >> "$LOG" 2>&1
rsync -az "$HOME/abag_xm/deepn/galaxy/" qb1:abag_xm/deepn/galaxy/ >> "$LOG" 2>&1
touch "$STATE"
echo "$(date -u) harvest+relay complete for $RUN" >> "$LOG"
"$HOME/.coworker/tg.sh" status "abag-xm deepn: $RUN drained on the galaxy; harvested + relayed to qb1 (rungs $RUNGS). Labelers pick the new folds up automatically; next pass can launch the next window phase."
