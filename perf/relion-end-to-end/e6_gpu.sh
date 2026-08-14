#!/bin/bash
# E6: the measured GPU arm, on a rented 2x H200 box.
#
# Same span (--continue from RELION's own run_it012_optimiser.star, to convergence), same 5x6 MPI
# shape, same flags as both qb1 arms, so the only variable against them is the hardware.
#
# Three arms on ONE box, and the third is the one that matters most:
#   g1  1x H200
#   g2  2x H200   -- two points give a MEASURED scaling slope instead of an assumed one
#   c   the same box's CPU, 5x6 like every other arm. This is what removes the "different host"
#       unknown that RELION's own job019 artifact could not: a within-machine GPU:CPU ratio is
#       directly comparable to our within-machine device:CPU ratio, where two absolute walls
#       measured on two different boxes are not.
#
# Stock RELION at e4a4aad, the same commit qb1 runs, built here from github rather than from our
# patched tree: a GPU reference a customer could reproduce must not carry our bridge or our
# instrument.
#
# Board power is sampled every 5 s for the watts column, so §7.2's per-watt row stops being a
# datasheet TDP on the GPU side.
set -u
R=/root/relion
D=/root/Tutorial5.0
COMMON="--continue $D/Refine3D/job019/run_it012_optimiser.star --j 6 --pool 6 \
  --dont_combine_weights_via_disc --preread_images"
mkdir -p /root/e6
cd "$D" || exit 1

run() {
  arm=$1; bin=$2; acc=$3
  [ -f /root/e6/$arm.done ] && return 0
  rm -f /root/e6/$arm.power
  nohup bash -c "while [ ! -f /root/e6/$arm.rc ]; do nvidia-smi --query-gpu=index,power.draw,utilization.gpu --format=csv,noheader >> /root/e6/$arm.power; sleep 5; done" > /dev/null 2>&1 < /dev/null &
  /usr/bin/time -f "%e %U %S %M" -o /root/e6/$arm.time \
    mpirun --allow-run-as-root -n 5 "$bin" --o /root/e6/${arm}_run $COMMON $acc \
    > /root/e6/$arm.log 2>&1
  echo "rc=$? arm=$arm acc=$acc" > /root/e6/$arm.rc
  echo done > /root/e6/$arm.done
}

run g1 "$R/build-gpu/bin/relion_refine_mpi" "--gpu 0"
run g2 "$R/build-gpu/bin/relion_refine_mpi" "--gpu 0:1"
run c  "$R/build-cpu/bin/relion_refine_mpi" "--cpu"
echo E6_DONE > /root/e6/campaign.done
