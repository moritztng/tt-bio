#!/usr/bin/env bash
# Ordered campaign for opendde-sizes-perf. Each leg takes benchlock in turn, so the legs serialise
# against each other and against any sibling worker's timed run. Legs are independent: a leg that
# fails does not stop the next one.
WT=/home/ttuser/.coworker/wt/opendde-sizes-perf
cd "$WT" || exit 1
BL=~/.coworker/scripts/benchlock.sh
export BENCHLOCK_WAIT_S=5400

# S1 first: the two shipped-regression checks at the two sizes where both levers can fire.
# `on` first and last is the A/A pair at each size.
$BL opendde-sizes-perf -- ./perf/sizes/run_probe.sh 128,256 on,nohbig,noout,on \
    perf/sizes/s1_128_256.json > perf/sizes/s1_128_256.log 2>&1
echo "S1 rc=$? $(date -Is)" >> perf/sizes/campaign.status

# S0 re-anchor at 512 on this card, and 768 Phase 0, in one process.
$BL opendde-sizes-perf -- ./perf/sizes/run_probe.sh 512,768 on,on \
    perf/sizes/anchor_512_768.json > perf/sizes/anchor_512_768.log 2>&1
echo "S0/768 rc=$? $(date -Is)" >> perf/sizes/campaign.status
