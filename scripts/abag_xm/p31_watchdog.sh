#!/bin/bash
# p31 window watchdog: on fleet completion (or safe deadline) respawn the prod fold worker
# pool the campaign killed to free the chips. The web tier, japanfold.service, cloudflared
# and SSH stay up the whole window, so restore = respawn the worker only -- NEVER
# maint-restore.sh, NEVER a cloudflared restart; that pair blips the tunnel Moritz's SSH
# rides (2026-08-04 lesson). Once the worker registers, online_workers goes non-zero, the
# platform's zero-device guard stops firing and the fold API serves again.
#
# The device lock is campaign-held across the p31+p32 chain; release it at the final drain
# (see state/abag-xm-deepn-n512.md).
#
# Worst case: 2600 chunk tasks, link gate tripped so every chunk folds fresh (~2100 card-h,
# ~85 h on 32 chips at the measured p29 fleet efficiency). Deadline 120 h covers that with
# margin; the expected window is ~64 h.
#
# p32 CHAIN (2026-08-08): the four formerly DRAM-excluded large targets fold at N=512 on
# the OOM-fixed engine tree as window p32, chained here: on a CLEAN p31 drain, launch
# p32_fleet.sh first and respawn prod only after P32_DONE (or its own 36 h deadline). On
# the deadline path (p31 never drained) skip p32 and respawn immediately. If p32 is not
# deployed yet at P31_DONE, respawn immediately -- never hold prod down waiting on a
# script that is not there.
B=$HOME/mthuening/p31
P32=$HOME/mthuening/p32
P32_SCRIPT=$HOME/mthuening/deepn_src_oomfix/scripts/abag_xm/p32_fleet.sh
DEADLINE=$(( $(date +%s) + 432000 ))   # 120h
CLEAN=0
while true; do
  sleep 600
  if grep -q P31_DONE $B/results.jsonl 2>/dev/null; then
    sleep 120   # let slot loops finish logging
    CLEAN=1
    break
  fi
  if [ $(date +%s) -gt $DEADLINE ]; then
    pgrep -f "abag_x[m]" > /dev/null || break   # deadline + no folds alive -> safe
  fi
done
if [ "$CLEAN" = 1 ] && [ -f "$P32_SCRIPT" ] && ! grep -q P32_DONE $P32/results.jsonl 2>/dev/null; then
  setsid nohup bash "$P32_SCRIPT" 32 8 >> $P32/fleet.log 2>&1 &
  echo "$(date -Is) p32 fleet launched pid=$! (large targets on oomfix engine)" >> $HOME/mthuening/p31_restore.log
  P32_DEADLINE=$(( $(date +%s) + 129600 ))   # 36h
  while true; do
    sleep 600
    if grep -q P32_DONE $P32/results.jsonl 2>/dev/null; then
      sleep 120
      break
    fi
    if [ $(date +%s) -gt $P32_DEADLINE ]; then
      pgrep -f "p32_flee[t]" > /dev/null || break
    fi
  done
fi
setsid nohup $HOME/mthuening/tt-bio/env/bin/tt-bio worker --connect http://127.0.0.1:8770 --accelerator tenstorrent >> $HOME/mthuening/prod_worker_restore.log 2>&1 &
echo "$(date -Is) prod worker respawned pid=$!" >> $HOME/mthuening/p31_restore.log
