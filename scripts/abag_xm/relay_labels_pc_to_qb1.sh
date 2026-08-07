#!/bin/bash
# Relay pc-produced labels.json (px/esm partition of the deepn galaxy tree) to qb1 every
# 10 min. pc labels into its harvest-staging tree; qb1 is where the analysis reads.
# One-directional pc -> qb1, no --delete, px/esm only: qb1 owns bz/od labeling, so the
# relay can never touch a bz/od label. Setsid-detached; kill by literal pid only.
set -u
LOG=$HOME/abag_xm/deepn/logs/relay_labels.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -u) relay armed (px/esm labels.json pc -> qb1, 600s)" >> "$LOG"
while :; do
  for m in protenix esmfold2; do
    rsync -az --include='*/' --include='labels.json' --exclude='*' \
      "$HOME/abag_xm/deepn/galaxy/$m/" "qb1:abag_xm/deepn/galaxy/$m/" >> "$LOG" 2>&1
  done
  sleep 600
done
