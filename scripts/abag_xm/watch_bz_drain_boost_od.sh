#!/bin/bash
# One-shot booster: when the bz galaxy tree has no unlabeled folds left, start a
# second od-dedicated labeler (2 workers) so bz's freed CPU shifts to the od long pole.
# Setsid-detached; exits after firing. Idempotent (pgrep guard). Kill-safety: the
# pgrep pattern never matches this script's own cmdline (label_od2 vs watch_bz).
set -u
G=/home/ttuser/abag_xm/deepn/galaxy/boltz2
WT=/home/ttuser/.coworker/wt/abag-xm-deepn-saturation-fullpanel
LOG=/home/ttuser/abag_xm/deepn/label_od2.log
echo "$(date -u) watcher armed" >> "$LOG"
while :; do
  pend=0
  for d in "$G"/*_n*/; do
    ls "$d"*_results_* >/dev/null 2>&1 || continue
    [ -f "$d/labels.json" ] || [ -f "$d/.label_lock" ] || pend=$((pend+1))
  done
  [ "$pend" = 0 ] && break
  sleep 300
done
if ! pgrep -f "deepn_label.py --workers 2 --base /home/ttuser/abag_xm/deepn/label_od2" >/dev/null; then
  mkdir -p /home/ttuser/abag_xm/deepn/label_od2
  ln -sfn ../galaxy/opendde /home/ttuser/abag_xm/deepn/label_od2/opendde
  cd "$WT"
  setsid nohup python3 -u scripts/abag_xm_deepn_label.py --workers 2 \
    --base /home/ttuser/abag_xm/deepn/label_od2 >> "$LOG" 2>&1 &
  echo "$(date -u) bz drained (0 pending) -> od booster labeler started (2 workers)" >> "$LOG"
fi
