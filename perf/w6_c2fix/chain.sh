#!/bin/bash
# Everything after the first sweep, in one unattended chain. Every stage is resumable, so a
# relaunch that finds this dead picks up where it stopped instead of repeating device time.
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
while pgrep -f "w6_c2fix/sweep.sh" >/dev/null; do sleep 20; done
echo "=== chain start $(date -u +%H:%M:%S) ==="
bash perf/w6_c2fix/paired.sh 4
bash perf/w6_c2fix/gate.sh BASE
bash perf/w6_c2fix/gate.sh C2FIX
bash perf/w6_c2fix/fpg.sh C2FIX
bash perf/w6_c2fix/fpg.sh BASE
/usr/bin/python3 perf/w6_c2fix/arm.py --arm C2FIX
echo "=== chain done $(date -u +%H:%M:%S) ==="
