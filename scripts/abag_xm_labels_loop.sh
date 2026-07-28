#!/usr/bin/env bash
# Keep the CPU-only labels campaign running so folds get labelled as Tier-A produces them.
#
# The campaign enumerates its task list once at startup, so a single run labels only the
# folds that existed when it began -- run it once and labelling stops there while
# generation carries on. This loop re-runs it, which is what keeps labelling overlapped
# with generation instead of piling ~30 h of it up at the end.
#
# Each run is capped at RUN_TIMEOUT: a stuck pairwise_matrix (9l1l once sat at 97 min and
# 0.02x CPU) is killed and the loop relaunches. Idempotent -- turn-38 fingerprinting skips
# any fold whose structures have not changed -- so nothing is redone.
#
#   Usage: scripts/abag_xm_labels_loop.sh [workers] [host_threads]
#          defaults 2 and 2, which is what is safe while this host is still folding.
#          Once the cards go idle, raise both (e.g. "8 4") to clear the backlog fast.
#
# CPU-only: never touches a device.
set -u
# Derive the worktree from this script, never a hardcoded path: the previous version
# pointed at the p3 slug, which is a concluded worktree -- fleet hygiene tears those down
# under a live job, and it also predates the --host_threads cap and the label fingerprint.
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ttuser/.abag_xm_label_venv/bin/python3
[ -x "$PY" ] || PY=/home/ttuser/tt-bio/env/bin/python3
WORKERS="${1:-2}"
HOST_THREADS="${2:-2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"
LOGDIR="$HOME/abag_xm/logs"
PROGRESS="$HOME/abag_xm/tier_a/progress.jsonl"
LABELS="$HOME/abag_xm/tier_a/labels"

mkdir -p "$LABELS" "$LOGDIR"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
count_ok(){ "$PY" -c "
import json,sys
p=sys.argv[1]
try: print(sum(1 for l in open(p) if json.loads(l).get('status')=='ok'))
except Exception: print(0)
" "$PROGRESS" 2>/dev/null || echo 0; }

log "START labels loop: wt=$WT workers=$WORKERS host_threads=$HOST_THREADS timeout=${RUN_TIMEOUT}s"
while true; do
  nlab=$(ls "$LABELS"/*.json 2>/dev/null | wc -l); nok=$(count_ok)
  log "labels=$nlab ok_folds=$nok — running campaign"
  timeout "$RUN_TIMEOUT" env PYTHONPATH="$WT" "$PY" "$WT/scripts/abag_xm_labels_campaign.py" \
      --workers "$WORKERS" --host_threads "$HOST_THREADS" >> "$LOGDIR/labels_campaign.log" 2>&1
  rc=$?
  nlab2=$(ls "$LABELS"/*.json 2>/dev/null | wc -l); nok2=$(count_ok)
  log "run exited rc=$rc labels $nlab->$nlab2 ok_folds $nok->$nok2"
  if [ "$nlab2" = "$nlab" ] && [ "$nok2" = "$nok" ]; then
    log "no new labels and no new ok folds — sleeping 600s"
    sleep 600
  else
    sleep 30
  fi
done
