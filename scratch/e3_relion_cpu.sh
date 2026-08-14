#!/bin/bash
# E3.4 -- the CPU RELION anchor, on OUR data, timed. Two arms:
#   arm A  one iteration from scratch: CurrentImageSize 32, the cheap early operating point
#   arm B  --continue from RELION own it012: CurrentImageSize 196, the expensive one that matters
# ALTCPU vectorised path (--cpu), 5 MPI x 6 threads, which is the tutorial own documented config.
S=/home/ttuser/.coworker/wt/relion-estep-integration/scratch
T=$S/Tutorial5.0
BIN=$S/relion/build/bin
export PATH=$BIN:$PATH
cd $T || exit 1
mkdir -p $S/cpurun

COMMON="--i Extract/job018/particles.star --ctf --particle_diameter 200 --flatten_solvent \
  --zero_mask --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 --sym D2 \
  --norm --scale --pad 1 --dont_combine_weights_via_disc --preread_images --pool 30 --cpu --j 6"

echo "=== ARM A: one iteration from scratch, ini_high 50 ==="
/usr/bin/time -v mpirun --allow-run-as-root -n 5 relion_refine_mpi \
  --o $S/cpurun/a_run $COMMON \
  --ref Class3D/job016/run_it025_class002_box256.mrc --firstiter_cc --ini_high 50 --iter 1 \
  > $S/cpurun/armA.log 2> $S/cpurun/armA.time
echo "armA rc=$?" | tee $S/cpurun/armA.rc

echo "=== ARM B: continue from it012, CurrentImageSize 196 ==="
/usr/bin/time -v mpirun --allow-run-as-root -n 5 relion_refine_mpi \
  --o $S/cpurun/b_run --continue $T/Refine3D/job019/run_it012_optimiser.star --iter 13 \
  --cpu --j 6 --pool 30 --dont_combine_weights_via_disc --preread_images \
  > $S/cpurun/armB.log 2> $S/cpurun/armB.time
echo "armB rc=$?" | tee $S/cpurun/armB.rc
echo ALLDONE > $S/cpurun/done
