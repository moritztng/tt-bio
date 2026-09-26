#!/bin/bash
# bcx-p10-rne leg 4: the fold A/B interleaved at the PROCESS boundary, four arms in one sitting.
#   ab.sh <rounds-per-process>
#
# Round-boundary alternation inside one process is the stronger design and this row started
# there, but it cannot run: the BindCraft 2 compile path DEADLOCKS AGAINST ITSELF at round 10 of
# this configuration, every time, on the same program hash. /proc/locks shows the process as both
# the holder and the waiter on `<xla-cache>/compile_497697782d79a246.lock`, 0 CPU ticks in 10 s,
# WCHAN `locks_lock_inode_wait`. It is not the private cache dir and not
# JAX_ENABLE_COMPILATION_CACHE: both runs died on the identical hash, one in the row's own dir
# and one in the host-global default.
#
# So each process stops at 9 rounds, short of the compile that hangs, and the arms alternate at
# the process boundary -- which is what the brief asked for. Order is off, fold, fold, off, so a
# monotonic drift in the box over the sitting cancels between the two arms instead of loading
# onto one. Round 1 of each process is its compile round and the report drops it, leaving 8
# timed rounds an arm.
set -euo pipefail
cd "$(dirname "$0")/../.."
rounds=${1:-9}
for arm in off fold fold off; do
    tag="p_${arm}_$(date +%H%M%S)"
    echo "### $(date -u +%H:%M:%SZ) arm=$arm rounds=$rounds tag=$tag"
    RNE_AB_ARMS=$arm perf/bcx_p10_rne/arm.sh "$tag" "$rounds" >"/tmp/rne_${tag}.log" 2>&1 \
        || { echo "arm $arm FAILED, see /tmp/rne_${tag}.log"; tail -20 "/tmp/rne_${tag}.log"; }
done
echo "### $(date -u +%H:%M:%SZ) all arms done"
ls -d perf/bcx_p10_rne/out/p_*/
