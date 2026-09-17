#!/usr/bin/env bash
cd /home/ttuser/.coworker/wt/c12-compose-fold || exit 1
setsid nohup ./perf/c12_compose/run.sh "$@" > /tmp/s2_launch.log 2>&1 < /dev/null &
echo "launched pid=$!"
