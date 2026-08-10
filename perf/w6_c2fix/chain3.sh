#!/bin/bash
# Follow-on to chain2: the one leg chain2's --legacy-rdx sweep could not score. Waits for chain2 to
# clear the card rather than contending with it -- one device context per card.
#
# Written as a SEPARATE script on purpose. chain2.sh is running, and bash reads a script lazily by
# byte offset: rewriting a script that is mid-execution makes the running instance resume at a
# shifted offset and die on a bogus syntax error. That is exactly what happened to fpg.sh's C2FIX
# instance at 00:01 (its gate had already completed, so no result was lost, but the `FPG DONE` line
# never printed). Never edit a running shell script; add a new one.
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
while pgrep -f "w6_c2fix/chain2.sh" >/dev/null; do sleep 60; done
echo "=== chain3 start $(date -u +%H:%M:%S) ==="
bash perf/w6_c2fix/boltz2env.sh C2FIX
bash perf/w6_c2fix/boltz2env.sh BASE
/usr/bin/python3 perf/w6_c2fix/arm.py --arm C2FIX
echo "=== chain3 done $(date -u +%H:%M:%S) ==="
