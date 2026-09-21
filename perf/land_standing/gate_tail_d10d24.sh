#!/bin/bash
# The tail of card 3's list, taken from the other end on card 2 so the two runners meet in the
# middle. The done_<arm> check at the top of each iteration keeps them from repeating work;
# the only race is two runners entering the same arm at once, which the device lease turns
# into a retryable failure rather than a wrong verdict.
WT=/home/ttuser/.coworker/wt/land-standing
cd "$WT" || exit 1
echo "tail runner, card 2, load ceiling 20, started $(date -u +%FT%TZ)" >> perf/land_standing/out/gate_d10d24/SUMMARY.txt
setsid ./perf/land_standing/gate_arms_d10d24.sh 2 20 protenix-v1 boltz2 protenix-v2 rfd3-fusion opendde esmfold2 esmfold2-fast </dev/null >/tmp/tail2.out 2>&1 &
sleep 1
exit 0
