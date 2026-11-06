#!/bin/bash
# Session 5's tail (orchestrator decision 11-04): after run_acc's Mol-1 exact 101-103, RFD4, then Mol-1 fast 101-103.
# The 104,105 seed sets are dropped as whole arms (WH graded 3 seeds). Takes over PVO-DONE from box_pvo.sh, whose
# own deadline (09:13Z) would have ended the session with RFD4 never run.
#   setsid nohup bash tail_pvo.sh <end_epoch> > tail_pvo.log 2>&1 &
set -u
END=${1:?end epoch}; B=/root/kc/pvo
say(){ echo "$(date -u +%FT%TZ) tail_pvo: $*"; }
# SIGKILL skips box_pvo.sh's exit path, so it never writes PVO-DONE; its children (smi logger, run_acc) live on.
pkill -9 -f '^bash box_pvo.sh' && say "box_pvo.sh stopped, run_acc and the smi logger keep running"
while pgrep -f run_acc.sh >/dev/null && [ ! -f $B/acc/mol1/exact_101-102-103/rc ]; do sleep 5; done
if pgrep -f run_acc.sh >/dev/null; then
  pkill -9 -f run_acc.sh; pkill -f 'mol1-venv/bin/(mol|python)'; sleep 5; pkill -9 -f 'mol1-venv/bin/(mol|python)'
  rm -rf $B/acc/mol1/fast_101-102-103; say "run_acc stopped after Mol-1 exact"
fi
rm -f $B/rfd4/RFD4-FAILED
# RFD4 keeps 90 min back for Mol-1 fast; its arms are bounded by work, the deadline is only a guard.
bash /root/kc/box_rfd4_pvo.sh $((END - 5400)) >> $B/extra.log 2>&1
bash /root/kc/run_acc.sh $END 101,102,103 >> $B/extra.log 2>&1
echo done > $B/PVO-DONE; say "PVO-DONE"
# FAILED 07:25Z: launched from a bare ssh shell, it had none of box_pvo.sh's exports (PATH with uv, MOL_ROOT_DIR),
# so RFD4 setup and Mol-1 fast both died in seconds and it still wrote PVO-DONE. Session 6 reran them via box_pvo.sh.
