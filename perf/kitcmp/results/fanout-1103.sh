#!/bin/bash
# Staggered 4-shard relaunch, 11-03. Each session logs to results/<label>.session.log.
cd /home/moritz/.coworker/wt/mol1-gpu-fp32/perf/kitcmp
for s in 0 1 2 3; do setsid nohup bash results/mol1-fp32-r$s/launch.sh > results/mol1-fp32-r$s/launcher.out 2>&1 < /dev/null & echo "r$s pid $!" >> results/fanout-1103.pids; sleep 75; done
