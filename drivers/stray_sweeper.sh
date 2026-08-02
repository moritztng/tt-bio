#!/bin/bash
# Kills llama3_1_8b_p150 test_pcc.py strays (deadlock by design, GH issue #17) that sibling
# agents keep relaunching on this shared host. Runs until sunrise file removed.
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
touch $WT/logs/SWEEPER_ACTIVE
while [ -f $WT/logs/SWEEPER_ACTIVE ]; do
  for pid in $(pgrep -f "llama3_1_8b_p150/tests/e2e/test_pcc.py" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null && echo "$(date -u) swept pid $pid" >> $WT/logs/sweeper.log
  done
  sleep 15
done
