#!/bin/bash
# p30 window watchdog: on fleet completion (or safe deadline) restore the prod fold
# worker the campaign killed to free chips (web tier + SSH stay up the whole window,
# so restore = respawn the worker only -- NEVER maint-restore.sh, NEVER a cloudflared
# restart; that pair blips the tunnel Moritz's SSH rides, 2026-08-04 lesson).
# Mirrors the proven p27b/p28 pattern. The device lock is campaign-held across windows
# (released at the FINAL window's drain, a checkpoint decision -- not here).
# Deadline bounds the worst case (link gate tripped -> all 2624 chunks fold fresh,
# ~825 card-h ~ 26h on 32 chips; the hardened runner caps any single hang at ~47min).
B=$HOME/mthuening/p30
DEADLINE=$(( $(date +%s) + 162000 ))   # 45h
while true; do
  sleep 600
  if grep -q P30_DONE $B/results.jsonl 2>/dev/null; then
    sleep 120   # let slot loops finish logging
    break
  fi
  if [ $(date +%s) -gt $DEADLINE ]; then
    pgrep -f "abag_x\[m\]" > /dev/null || break   # deadline + no folds alive -> safe
  fi
done
setsid nohup $HOME/mthuening/tt-bio/env/bin/tt-bio worker --connect http://127.0.0.1:8770 --accelerator tenstorrent >> $HOME/mthuening/prod_worker_restore.log 2>&1 &
echo "$(date -Is) prod worker respawned pid=$!" >> $HOME/mthuening/p30_restore.log
