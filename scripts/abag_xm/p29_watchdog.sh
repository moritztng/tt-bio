#!/bin/bash
# p29 window watchdog: on fleet completion (or safe deadline) restore the prod fold
# worker the campaign killed to free chips (web tier + SSH stay up the whole window,
# so restore = respawn the worker only -- NEVER maint-restore.sh, NEVER a cloudflared
# restart; that pair blips the tunnel Moritz's SSH rides, 2026-08-04 lesson).
# Mirrors the proven p27b pattern. The device lock is campaign-held across windows;
# if p29 is the campaign's FINAL device window, release the lock at this drain
# (checkpoint decision -- see the state doc), not inside this script.
# Worst case: 1304 px/esm N=256 chunk tasks, link gate tripped (chunk-0 folds fresh),
# ~450 card-h ~ 15h on 32 chips; deadline 30h covers 2x with the hardened runner.
B=$HOME/mthuening/p29
DEADLINE=$(( $(date +%s) + 108000 ))   # 30h
while true; do
  sleep 600
  if grep -q P29_DONE $B/results.jsonl 2>/dev/null; then
    sleep 120   # let slot loops finish logging
    break
  fi
  if [ $(date +%s) -gt $DEADLINE ]; then
    pgrep -f "abag_x\[m\]" > /dev/null || break   # deadline + no folds alive -> safe
  fi
done
setsid nohup $HOME/mthuening/tt-bio/env/bin/tt-bio worker --connect http://127.0.0.1:8770 --accelerator tenstorrent >> $HOME/mthuening/prod_worker_restore.log 2>&1 &
echo "$(date -Is) prod worker respawned pid=$!" >> $HOME/mthuening/p29_restore.log
