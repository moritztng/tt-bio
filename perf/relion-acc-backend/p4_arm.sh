#!/bin/bash
# Arm T2: same as P3 arm T, with the bridge mutex scoped to initialisation instead of the whole call.
# Reference is P3 arm A, already on disk, so only this arm needs re-running.
set -u
S=/home/ttuser/relion-scratch
cd $S/Tutorial5.0 || exit 1
uptime > $S/p4/t2.loadavg
/usr/bin/time -f "%e %M" -o $S/p4/t2.time \
  mpirun -n 3 -x PYTHONPATH=/home/ttuser/.coworker/wt/relion-acc-backend -x TT_RELION_BACKEND=torch \
    $S/relion/build-tt/bin/relion_refine_mpi --o $S/p4/t2_run \
    --continue $S/Tutorial5.0/Refine3D/job019/run_it012_optimiser.star --auto_iter_max 13 \
    --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images > $S/p4/t2.log 2>&1
echo "t2 rc=$?" > $S/p4/rc
