#!/usr/bin/env bash
# Leg 3, separate file because campaign.sh is still executing and bash resumes a script by byte
# offset -- editing a running script in place corrupts it (memory never-edit-a-running-shell-script).
# Queues on benchlock behind leg 2, so it starts by itself once 768 finishes.
WT=/home/ttuser/.coworker/wt/opendde-sizes-perf
cd "$WT" || exit 1
export BENCHLOCK_WAIT_S=5400
~/.coworker/scripts/benchlock.sh opendde-sizes-perf -- ./perf/sizes/run_probe.sh 512 on,wbig,on \
    perf/sizes/wbig_512.json > perf/sizes/wbig_512.log 2>&1
echo "wbig512 rc=$? $(date -Is)" >> perf/sizes/campaign.status
