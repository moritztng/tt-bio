#!/usr/bin/env bash
# Snapshot per-fold DeepRank JSONs out of the live scorer's work dirs into
# ~/abag_xm/deeprank_harvest/ once a minute, so a crash or reboot loses the process but not
# the scores. Insurance for runs launched before the stable-cache fix in
# abag_xm_ranker_scores.py (_run_deeprank_batched writes straight to a persistent cache now);
# once every in-flight leg has exited, this script has no further purpose.
set -u
DEST="$HOME/abag_xm/deeprank_harvest"
LOG="$HOME/abag_xm/logs/deeprank_harvest.log"
mkdir -p "$DEST" "$(dirname "$LOG")"
echo "[$(date -u +%FT%TZ)] harvester start pid $$ on $(hostname -s)" >> "$LOG"
deadline=$(( $(date +%s) + 86400 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  for src in /tmp/deeprank_manifest_*/ "$HOME/abag_xm/tier_a/deeprank_json_cache"/; do
    [ -d "$src" ] || continue
    b=$(basename "$src")
    mkdir -p "$DEST/$b"
    cp -u "$src"*.json "$DEST/$b/" 2>/dev/null
  done
  cp -u "$HOME/abag_xm/tier_a/ranker_scores.csv" \
        "$DEST/ranker_scores.csv.$(hostname -s).snapshot" 2>/dev/null
  if ! pgrep -f "abag_xm_ranker_scores.py" >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] no ranker_scores driver alive; final pass complete, exiting" >> "$LOG"
    exit 0
  fi
  sleep 60
done
echo "[$(date -u +%FT%TZ)] 24h deadline reached, exiting" >> "$LOG"
