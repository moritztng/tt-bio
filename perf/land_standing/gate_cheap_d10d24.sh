#!/bin/bash
WT=/home/ttuser/.coworker/wt/land-standing
cd "$WT" || exit 1
echo "cheap runner, card 1, load ceiling 20, started $(date -u +%FT%TZ)" >> perf/land_standing/out/gate_d10d24/SUMMARY.txt
setsid ./perf/land_standing/gate_arms_d10d24.sh 1 20 nesso1 pxdesign esmc-300m esmc-600m </dev/null >/tmp/cheap1.out 2>&1 &
sleep 1
exit 0
