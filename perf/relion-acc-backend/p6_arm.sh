#!/bin/bash
# Matched-load reference for arm T2: same command as P3 arm A, run now, so the T2 wall has a
# same-conditions partner. P3 arm A ran at loadavg 22.42, T2 at 1.24, so the P3 pair is not
# comparable and this closes that hole. Doubles as an A/A against P3 arm A at a different load.
set -u
S=/home/ttuser/relion-scratch
cd $S/Tutorial5.0 || exit 1
uptime > $S/p6/t3.loadavg
BL=/home/ttuser/.coworker/scripts/benchlock.sh
export BENCHLOCK_WAIT_S=600
$BL worker:relion-acc-backend -- /usr/bin/time -f "%e %M" -o $S/p6/t3.time \
  mpirun -n 3 -x PYTHONPATH=/home/ttuser/.coworker/wt/relion-acc-backend -x TT_RELION_BACKEND=torch \
    $S/relion/build-tt/bin/relion_refine_mpi --o $S/p6/t3_run \
    --continue $S/Tutorial5.0/Refine3D/job019/run_it012_optimiser.star --auto_iter_max 13 \
    --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images > $S/p6/t3.log 2>&1
echo "a2 rc=$?" > $S/p6/rc
uptime > $S/p6/t3.loadavg_end
