#!/bin/bash
# The two folds section 40.5 says are owed. Both are fidelity, so they need no quiet box,
# but on this host every window is inside someone's benchlocked campaign, so the only way
# to fold without contaminating a sibling's measurement is to hold the lock ourselves.
# Scoring is host CPU and runs outside the lock.
WT=/home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
PY=/home/ttuser/tt-bio-dev/env/bin/python3
LOCK=$HOME/.coworker/scripts/benchlock.sh
cd $WT
export BENCHLOCK_WAIT_S=3000 BENCHLOCK_LOAD_WAIT_S=120

echo "##### hema_512 lever-off arm, queued $(date -u +%H:%M:%S)"
$LOCK openbind-fused-fidelity -- bash perf/obfused/disto512_of3.sh 1 0,1,2 hema_512
echo "##### hema done $(date -u +%H:%M:%S)"

# Score the four-target pool while the anchor waits for the lock again.
setsid nohup env PYTHONPATH=$WT $PY -u perf/fused_sdpa/disto_multi.py --rung 512 \
    --dir perf/obfused/disto512 --out perf/obfused/disto512_k4.json \
    > perf/obfused/logs/score_k4.log 2>&1 < /dev/null &

echo "##### 9bk6 anchor, queued $(date -u +%H:%M:%S)"
$LOCK openbind-fused-fidelity -- bash perf/obfused/anchor9bk6.sh 1
echo "##### anchor done $(date -u +%H:%M:%S)"
