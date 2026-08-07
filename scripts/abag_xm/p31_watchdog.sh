#!/bin/bash
# p31 window watchdog: on fleet completion (or safe deadline) respawn the prod fold worker
# pool the campaign killed to free the chips. The web tier, japanfold.service, cloudflared
# and SSH stay up the whole window, so restore = respawn the worker only -- NEVER
# maint-restore.sh, NEVER a cloudflared restart; that pair blips the tunnel Moritz's SSH
# rides (2026-08-04 lesson). Once the worker registers, online_workers goes non-zero, the
# platform's zero-device guard stops firing and the fold API serves again.
#
# The device lock is campaign-held; p31 is this campaign's only device window, so release
# it at this drain (see state/abag-xm-deepn-n512.md).
#
# Worst case: 2600 chunk tasks, link gate tripped so every chunk folds fresh (~2100 card-h,
# ~85 h on 32 chips at the measured p29 fleet efficiency). Deadline 120 h covers that with
# margin; the expected window is ~64 h.
B=$HOME/mthuening/p31
DEADLINE=$(( $(date +%s) + 432000 ))   # 120h
while true; do
  sleep 600
  if grep -q P31_DONE $B/results.jsonl 2>/dev/null; then
    sleep 120   # let slot loops finish logging
    break
  fi
  if [ $(date +%s) -gt $DEADLINE ]; then
    pgrep -f "abag_x[m]" > /dev/null || break   # deadline + no folds alive -> safe
  fi
done
setsid nohup $HOME/mthuening/tt-bio/env/bin/tt-bio worker --connect http://127.0.0.1:8770 --accelerator tenstorrent >> $HOME/mthuening/prod_worker_restore.log 2>&1 &
echo "$(date -Is) prod worker respawned pid=$!" >> $HOME/mthuening/p31_restore.log
