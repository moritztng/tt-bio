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
#          Once the cards go idle the loop raises itself to IDLE_WORKERS x
#          IDLE_HOST_THREADS (default nproc/2 x 2) and drops back if folding resumes, so
#          neither setting has to be changed by hand mid-campaign.
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
# Concurrency to use once the cards go idle. The header used to say "raise both by hand once the
# cards go idle", and that is a schedule improvement nobody is present to make: measured on qb2,
# its 113-fold backlog is 11.7 core-hours = 5.8 h at 2 workers, against ~4 h of generation left,
# so labelling becomes the critical path and then runs ~2 h past the last fold at the timid
# setting. Sized to the box rather than hardcoded, and re-evaluated every iteration below.
IDLE_WORKERS="${IDLE_WORKERS:-$(( $(nproc) / 2 ))}"
IDLE_HOST_THREADS="${IDLE_HOST_THREADS:-2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"
LOGDIR="$HOME/abag_xm/logs"
PROGRESS="$HOME/abag_xm/tier_a/progress.jsonl"
LABELS="$HOME/abag_xm/tier_a/labels"

mkdir -p "$LABELS" "$LOGDIR"
# ISO8601 with the offset: these hosts are UTC and the orchestrating side is not, so a bare
# wall clock here is two hours out of step with whoever reads it.
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] $*"; }
count_ok(){ "$PY" -c "
import json,sys
p=sys.argv[1]
try: print(sum(1 for l in open(p) if json.loads(l).get('status')=='ok'))
except Exception: print(0)
" "$PROGRESS" 2>/dev/null || echo 0; }

log "START labels loop: wt=$WT busy=${WORKERS}x${HOST_THREADS} idle=${IDLE_WORKERS}x${IDLE_HOST_THREADS} timeout=${RUN_TIMEOUT}s"
while true; do
  nlab=$(ls "$LABELS"/*.json 2>/dev/null | wc -l); nok=$(count_ok)
  # Re-checked every iteration, not once at startup: the whole point is to notice the cards
  # draining hours after this loop began. Same predicate the endgame uses to gate DeepRank-Ab.
  if pgrep -f "tt_bio.mai[n] predict" >/dev/null 2>&1; then
    w="$WORKERS"; ht="$HOST_THREADS"; regime="cards busy"
  else
    w="$IDLE_WORKERS"; ht="$IDLE_HOST_THREADS"; regime="cards IDLE"
  fi
  log "labels=$nlab ok_folds=$nok — running campaign ($regime: workers=$w host_threads=$ht)"
  # PYTHONUNBUFFERED + -u because the campaign log is the only live signal for a stage whose
  # items take 6-21 minutes. Block-buffered, the log sits stale for a quarter of an hour while
  # work proceeds normally, which is indistinguishable from a hang -- on 2026-07-28 that cost a
  # pass trying to tell whether qb2 was stalled or just slow. The generate driver already does
  # this; the label loop did not.
  timeout "$RUN_TIMEOUT" env PYTHONPATH="$WT" PYTHONUNBUFFERED=1 "$PY" -u \
      "$WT/scripts/abag_xm_labels_campaign.py" \
      --workers "$w" --host_threads "$ht" >> "$LOGDIR/labels_campaign.log" 2>&1
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
