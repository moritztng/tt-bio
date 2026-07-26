#!/usr/bin/env bash
# abag_xm_labels_loop.sh — continuously runs the CPU-only labels campaign so new
# ok folds get labeled as Tier-A completes. Each campaign run is capped at
# RUN_TIMEOUT (1800s=30min): a stuck pairwise_matrix (e.g. 9l1l hung at 97min
# 0.02x CPU) gets killed and the loop relaunches; idempotency skips already-
# labeled pairs so no work is redone. 9l1l will retry each loop; if it keeps
# hanging, label it manually or skip. CPU-only — does NOT touch the device.
set +u
WT=/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p3
PY=/home/ttuser/.abag_xm_label_venv/bin/python3
RUN_TIMEOUT=1800
WORKERS=6
log(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$HOME/abag_xm/tier_a/labels"
log "START labels loop: workers=$WORKERS run_timeout=${RUN_TIMEOUT}s"
while true; do
  nlab=$(ls "$HOME/abag_xm/tier_a/labels"/*.json 2>/dev/null | wc -l)
  nok=$(python3 -c "import json;print(sum(1 for l in open(\"$HOME/abag_xm/tier_a/progress.jsonl\") if json.loads(l).get(\"status\")==\"ok\"))" 2>/dev/null || echo 0)
  log "labels=$nlab ok=$nok — running campaign (timeout ${RUN_TIMEOUT}s)"
  timeout "$RUN_TIMEOUT" env PYTHONPATH="$WT" "$PY" "$WT/scripts/abag_xm_labels_campaign.py" --workers "$WORKERS" >> /tmp/labels_campaign.log 2>&1
  rc=$?
  nlab2=$(ls "$HOME/abag_xm/tier_a/labels"/*.json 2>/dev/null | wc -l)
  log "campaign run exited rc=$rc labels $nlab->$nlab2"
  # if no progress and no new ok folds, slow the loop
  nok2=$(python3 -c "import json;print(sum(1 for l in open(\"$HOME/abag_xm/tier_a/progress.jsonl\") if json.loads(l).get(\"status\")==\"ok\"))" 2>/dev/null || echo 0)
  if [ "$nlab2" = "$nlab" ] && [ "$nok2" = "$nok" ]; then
    log "no new labels AND no new ok folds — sleeping 600s before retry"
    sleep 600
  else
    sleep 30
  fi
done
