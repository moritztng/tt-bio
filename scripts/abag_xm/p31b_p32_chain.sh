#!/bin/bash
# Replacement for p31_watchdog.sh's p32 chain, which cannot fire.
#
# WHY THIS EXISTS (pass 21, 2026-08-10). p31_watchdog.sh gates the p32 launch on
# `grep -q P31_DONE p31/results.jsonl`. That marker is written by a FIRST-instance
# p31_fleet.sh at its line 510. The original p31 driver died without writing it, no
# p31_fleet.sh has been alive since, and the surviving driver is p31b_fleet.sh, whose
# launch environment is DONE_MARK=P31B_DONE / DONE_FILE=p31/p31b.done -- it never touches
# results.jsonl. So CLEAN stays 0 forever, the p32 block is skipped outright, and the old
# watchdog just sits until its 120 h deadline (2026-08-13T06:14:50Z) and respawns prod.
#
# That is not a cosmetic miss. p32's 48 cells are exactly the ones p31's tasks.txt omits:
#   opendde-abag  9i3p 9j4c 9ivj 9q7y  x c0-7 = 32
#   protenix-v2   9j4c                 x c0-7 =  8
#   esmfold2      9j4c                 x c0-7 =  8
# Without p32 the 512 panel ships with the same large-target holes the campaign was paused
# on 2026-08-07 to close, silently, with every liveness signal green.
#
# TRIGGER. Not a marker: the absence of the fold fleet. p31b_fleet.sh exits when it has
# walked its task index, and pass 18 licenses further p31b generations, so a marker-based
# trigger either misses (generation 1's marker never arrives) or false-fires (generation
# 1's marker is still there when generation 2 starts). Two consecutive idle polls 300 s
# apart is the condition that is true exactly when no fold fleet is running.
#
# CHIPS. The full 27-chip quarantine set (excl. 0 2 10 16 18). p31b's 22 was the idle set
# at 18:02Z, derived by subtracting live folds' TT_VISIBLE_DEVICES from the 27, not a
# health judgement, so 9 13 15 17 21 come back when p31b exits. p32's CHIPS list is fixed
# at launch, so parallelism is decided once: 48 cells over 27 slots is 2 waves (~7-12 h),
# over 22 is 3 (~11-18 h), over the 5 idle right now is 10 (~35-58 h). Serial after p31b.
#
# The device lock is released only after prod is folding again -- at that point no campaign
# process needs a chip, and leaving it held blocks the rest of the fleet.
set -u
M=$HOME/mthuening
B=$M/p31
P32=$M/p32
P32_SCRIPT=$M/deepn_src_oomfix/scripts/abag_xm/p32_fleet.sh
LOG=$M/p31_restore.log
CHIPS_SET="1 3 4 5 6 7 8 9 11 12 13 14 15 17 19 20 21 22 23 24 25 26 27 28 29 30 31"
WAIT_CAP=$(( $(date +%s) + 172800 ))   # 48 h: p31b wedged beyond any guard -> restore prod, skip p32

say() { echo "$(date -Is) chain: $*" >> "$LOG"; }

# Matched on argv POSITION, not by substring. `pgrep -f p31b_fleet.sh` also matches any
# `bash -c '... p31b_fleet.sh ...'` wrapper, and an ssh one-liner that merely mentions the
# driver path is enough to make the fleet look alive -- measured, it hung the first draft
# of this script indefinitely. A real driver is argv[0]=bash argv[1]=<path>/p31b_fleet.sh;
# a wrapper's argv[1] is "-c". Same for the folds: argv[0] is the python binary.
fleet_alive() {
  ps -eo args= | awk '
    $1 ~ /(^|\/)bash$/   && $2 ~ /p31b?_fleet\.sh$/       { f = 1 }
    $1 ~ /(^|\/)python/  && /tt_bio\.main predict/ && /mthuening/ { f = 1 }
    END { exit !f }'
}

say "armed (pid $$); waiting for the p31b fleet to exit"
idle=0
while true; do
  sleep 300
  if fleet_alive; then
    idle=0
  else
    idle=$((idle + 1))
    say "fleet idle poll $idle/2"
    [ $idle -ge 2 ] && break
  fi
  if [ "$(date +%s)" -gt $WAIT_CAP ]; then
    say "WAIT CAP hit with the fleet still alive -- skipping p32, restoring prod"
    idle=-1; break
  fi
done

if [ $idle -ge 2 ] && [ -f "$P32_SCRIPT" ] && ! grep -q P32_DONE "$P32/results.jsonl" 2>/dev/null; then
  say "launching p32 on 27 chips (48 large-target cells, oomfix engine $(cat $P32/engine_commit.txt 2>/dev/null))"
  CHIPS="$CHIPS_SET" setsid nohup bash "$P32_SCRIPT" 27 8 </dev/null >> "$P32/fleet.log" 2>&1 &
  say "p32 launched pid=$!"
  P32_DEADLINE=$(( $(date +%s) + 129600 ))   # 36 h, as the original chain
  while true; do
    sleep 600
    if grep -q P32_DONE "$P32/results.jsonl" 2>/dev/null; then
      sleep 120   # let the slot loops finish logging
      say "P32_DONE"
      break
    fi
    if [ "$(date +%s)" -gt $P32_DEADLINE ]; then
      pgrep -f "p32_flee[t]" >/dev/null || { say "p32 deadline reached, no driver alive"; break; }
    fi
  done
else
  say "p32 skipped (idle=$idle, script $( [ -f "$P32_SCRIPT" ] && echo present || echo MISSING ))"
fi

setsid nohup $M/tt-bio/env/bin/tt-bio worker --connect http://127.0.0.1:8770 \
  --accelerator tenstorrent </dev/null >> "$M/prod_worker_restore.log" 2>&1 &
say "prod worker respawned pid=$!"
sleep 60
# `release` takes the owner and refuses on a mismatch, so this is a no-op if someone else
# has taken the lock in the meantime. Owner string is the one held since 2026-08-08T05:43Z.
bash "$M/galaxy_device_lock.sh" release abag-xm-deepn-n512-p31 >> "$LOG" 2>&1 \
  && say "device lock released"
