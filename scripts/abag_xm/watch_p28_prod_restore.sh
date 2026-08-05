#!/bin/bash
# pc-side p28 prod-restore FALLBACK monitor. The galaxy-side p28_watchdog.sh is the
# primary restore leg (respawn the prod fold worker at P28_DONE); two detached
# galaxy-side ancestors of that watchdog died silently (passes 232/233), so this
# independent pc-side layer polls the same sentinel over ssh. On P28_DONE it waits
# up to 15 min for the primary to fire; only if no prod worker appears does it
# spawn one itself (idempotent pgrep guard on the remote side). NEVER touches
# japanfold.service, cloudflared, or sshd. Absolute deadline mirrors the primary's
# 48h bound (p28 ~20h window + margin); at deadline it restores only when no fold processes remain
# (and keeps waiting on ssh failure or live folds -- never restores mid-fleet).
SSH="ssh -o ConnectTimeout=15 -o BatchMode=yes japanfold-ssh"
DEADLINE=$(( $(date +%s) + 172800 ))  # 48h: p28 ~20h window + margin
LOG=$HOME/.coworker/state/p28_prod_restore_monitor.log
RESULTS=/home/cust-team/mthuening/p28/results.jsonl

while true; do
  sleep 600
  if $SSH "grep -q P28_DONE $RESULTS" 2>/dev/null; then
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    $SSH 'pgrep -f "abag_x[m]" > /dev/null' 2>/dev/null
    [ $? -eq 1 ] && break   # ssh ok + no folds alive -> safe
  fi
done

echo "$(date -Is) sentinel seen; waiting up to 900s for the primary to fire" >> "$LOG"

# Reap leaked spawn_main orphans (pass-239 leak class: timeout TERM-kills the fold
# main at the 21600s cap; its spawn grandchild reparents to pid 1 and keeps holding
# the chip) BEFORE any prod-worker spawn leg fires, so prod never collides with a
# held devnode. ppid=1 + spawn_main filter only matches dead folds' grandchildren;
# kill -9 because they ignore TERM. Idempotent and safe post-sentinel (all p28 fold
# mains are dead by definition of P28_DONE).
$SSH 'PIDS=$(ps -eo pid,ppid,cmd | grep spawn_main | grep -v grep | awk "\$2==1 {print \$1}"); [ -n "$PIDS" ] && { echo "$PIDS" | while read -r P; do kill -9 "$P" 2>/dev/null && echo "reaped $P"; done; }; exit 0' >> "$LOG" 2>&1
echo "$(date -Is) orphan reap leg done (remote rc=$?)" >> "$LOG"

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
