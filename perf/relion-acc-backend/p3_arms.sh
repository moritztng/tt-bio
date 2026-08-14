#!/bin/bash
# P3 -- one RELION iteration, twice, on the same binary, differing only in whether the coarse
# compare goes through the bridge. Identical binary means the A/B has exactly one variable.
#
#   arm A  TT_RELION_BACKEND=ttnn   the bridge declines every call, so RELION runs its own kernel
#   arm T  TT_RELION_BACKEND=torch  the bridge handles every coarse call
#
# --auto_iter_max 13 stops after one iteration; --iter alone does not, auto-refine ignores it.
# --cpu is mandatory: without it RELION never enters src/acc and the bridge is never called.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
WT=/home/ttuser/.coworker/wt/relion-acc-backend
BIN=$S/relion/build-tt/bin/relion_refine_mpi
mkdir -p $S/p3
rm -f $S/p3/rc $S/p3/done
cd $T || exit 1

COMMON="--continue $T/Refine3D/job019/run_it012_optimiser.star --auto_iter_max 13 \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"

for arm in a t; do
  [ $arm = a ] && BK=ttnn || BK=torch
  uptime > $S/p3/$arm.loadavg
  /usr/bin/time -f "%e %M" -o $S/p3/$arm.time \
    mpirun -n 3 -x PYTHONPATH=$WT -x TT_RELION_BACKEND=$BK \
      $BIN --o $S/p3/${arm}_run $COMMON > $S/p3/$arm.log 2>&1
  echo "$arm rc=$? backend=$BK" >> $S/p3/rc
done
echo P3_DONE > $S/p3/done
