#!/bin/bash
# pc-side p27 prod-restore FALLBACK monitor. The galaxy-side p27b_watchdog.sh is the
# primary restore leg (respawn the prod fold worker at P27_DONE); two detached
# galaxy-side ancestors of that watchdog died silently (passes 232/233), so this
# independent pc-side layer polls the same sentinel over ssh. On P27_DONE it waits
# up to 15 min for the primary to fire; only if no prod worker appears does it
# spawn one itself (idempotent pgrep guard on the remote side). NEVER touches
# japanfold.service, cloudflared, or sshd. Absolute deadline mirrors the primary's
# original 30h bound; at deadline it restores only when no fold processes remain
# (and keeps waiting on ssh failure or live folds -- never restores mid-fleet).
SSH="ssh -o ConnectTimeout=15 -o BatchMode=yes japanfold-ssh"
DEADLINE=1785984000   # 2026-08-06 02:40 UTC
LOG=$HOME/.coworker/state/p27_prod_restore_monitor.log
RESULTS=/home/cust-team/mthuening/p27/results.jsonl

while true; do
  sleep 600
  if $SSH "grep -q P27_DONE $RESULTS" 2>/dev/null; then
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    $SSH 'pgrep -f "abag_x[m]" > /dev/null' 2>/dev/null
    [ $? -eq 1 ] && break   # ssh ok + no folds alive -> safe
  fi
done

echo "$(date -Is) sentinel seen; waiting up to 900s for the primary to fire" >> "$LOG"
for _ in 1 2 3 4 5 6 7 8 9; do
  sleep 100
  if $SSH 'pgrep -f "tt-bio worker --connect" > /dev/null' 2>/dev/null; then
    echo "$(date -Is) primary fired (prod worker alive); fallback standing down" >> "$LOG"
    exit 0
  fi
done

echo "$(date -Is) primary absent after 900s; fallback spawning prod worker" >> "$LOG"
$SSH 'pgrep -f "tt-bio worker --connect" > /dev/null && exit 0; setsid nohup /home/cust-team/mthuening/tt-bio/env/bin/tt-bio worker --connect http://127.0.0.1:8770 --accelerator tenstorrent >> /home/cust-team/mthuening/prod_worker_restore.log 2>&1 < /dev/null & exit 0' >> "$LOG" 2>&1
echo "$(date -Is) fallback restore leg done (remote rc=$?)" >> "$LOG"
