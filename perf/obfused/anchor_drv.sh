#!/bin/bash
# The last owed fold: the 9bk6_164 three-arm anchor, then its two-pair scoring.
# hema_512 is already complete (both arms folded), so finish2.sh's first block is a no-op
# and this driver skips straight to the anchor. Long benchlock wait on purpose: a detached
# waiter outlives the worker turn that queued it, and on this host the lock is held
# continuously by long benchmark campaigns.
WT=/home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
LOCK=$HOME/.coworker/scripts/benchlock.sh
cd $WT
export BENCHLOCK_WAIT_S=7200 BENCHLOCK_LOAD_WAIT_S=180
echo "##### 9bk6 anchor queued $(date -u +%H:%M:%S)"
$LOCK openbind-fused-anchor -- bash perf/obfused/anchor9bk6.sh 1
echo "##### anchor folds done $(date -u +%H:%M:%S)"
bash perf/obfused/score_anchor.sh
echo "##### anchor scored $(date -u +%H:%M:%S)"
