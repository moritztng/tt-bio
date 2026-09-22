#!/bin/bash
# The cheap arms, on card 2. Card 1 still holds the orphaned `capacity` arm, and tt-bio's
# device lease refuses a second open of the same card by the same holder identity, which is
# what failed pxdesign at 20:38Z: an infrastructure collision that the gate reports as
# "GATE FAIL", indistinguishable in SUMMARY.txt from a real accuracy failure.
WT=/home/ttuser/.coworker/wt/land-standing
cd "$WT" || exit 1
echo "cheap runner, card 2, load ceiling 20, started $(date -u +%FT%TZ)" >> perf/land_standing/out/gate_d10d24/SUMMARY.txt
setsid ./perf/land_standing/gate_arms_d10d24.sh 2 20 pxdesign esmc-300m esmc-600m </dev/null >/tmp/cheap2.out 2>&1 &
sleep 1
exit 0
