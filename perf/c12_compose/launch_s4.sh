#!/usr/bin/env bash
# Session s4: the confirmation replicate for s3, and it answers ONE question at high power.
#
# s3 measured all three arms and its subadditivity term at n=31, so the singles and the interaction
# are already in hand. What a replicate adds is independence: a second process, a second model load,
# and a box running at loadavg ~2 instead of the 2.4-9.7 s3 saw while the release gate's size ladder
# climbed. So s4 drops the two single-lever arms and spends every fold on base vs both, which puts
# reps into the composed delta at twice the rate: 3 folds per rep instead of 5.
#
# base sits at pos0 and pos2 with `both` between them, so `both` is differenced against the mean of
# the two bases that bracket it and a linear within-rep drift cancels exactly. --palindrome is not
# passed because a one-arm interior has nothing to reverse.
cd /home/ttuser/.coworker/wt/c12-compose-fold || exit 1
MAXLOAD=6.0 setsid nohup ./perf/c12_compose/run.sh s4 \
  --reps 26 --arms base,both,base \
  > perf/c12_compose/out/s4_launch.log 2>&1 < /dev/null &
echo "s4 launched pid=$! at $(date -u +%FT%TZ)"
