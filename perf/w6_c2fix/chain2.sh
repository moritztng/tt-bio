#!/bin/bash
# Everything re-measured on main@f7c485c5, after E7's batched_matmul merge moved the baseline
# again. Every stage skips work whose output already exists, so a relaunch that finds this dead
# resumes instead of repeating device time.
#
# Order is deliberate. A sibling perfwar leg is running its own full_parity_gate on card 3, and
# W11 measured that concurrent gates on this box contaminate TIMING but not correctness (a 117 aa
# fold went 45 s -> 6 min under a sibling gate, same card, same arm). So the correctness legs run
# first and soak up the contended window, and the timing rounds are spread around them: one round
# early for an immediate ratio, the rest after the gates, when the box has likely drained.
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
echo "=== chain2 start $(date -u +%H:%M:%S) on main@f7c485c5 ==="
bash perf/w6_c2fix/paired.sh 1
bash perf/w6_c2fix/fpg.sh C2FIX
bash perf/w6_c2fix/fpg.sh BASE
bash perf/w6_c2fix/gate.sh C2FIX
bash perf/w6_c2fix/gate.sh BASE
bash perf/w6_c2fix/paired.sh 3
bash perf/w6_c2fix/small.sh
/usr/bin/python3 perf/w6_c2fix/arm.py --arm C2FIX
echo "=== chain2 done $(date -u +%H:%M:%S) ==="
