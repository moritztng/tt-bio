#!/bin/bash
# The rented GPU ladder for the production-scale arm.
#
# `relion-end-to-end` §7 measured one point per arm on a 4,452-particle job and lost to a single
# H200. Whether that survives at 111,300 particles is a question about SLOPES, so every arm here
# runs the same single iteration at several particle counts on ONE box: a within-machine ladder,
# which is what makes a within-machine device:CPU ratio comparable to ours.
#
# Stock RELION at the same commit qb1 runs, built here rather than shipped from our patched tree,
# so the GPU reference carries neither our bridge nor our instrument.
#
# Cheapest point first, and every arm writes a done-marker, so a box that has to be destroyed
# early still leaves a fittable ladder behind.
set -u
R=/root/relion
D=/root/Tutorial5.0
O=/root/prod
mkdir -p "$O"
COMMON="--auto_iter_max 13 --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"
cd "$D" || exit 1

run() {
  arm=$1; bin=$2; acc=$3; k=$4
  [ -f "$O/$arm.done" ] && { echo "skip $arm"; return 0; }
  rm -f "$O/$arm.power" "$O/$arm.rc"
  nohup bash -c "while [ ! -f $O/$arm.rc ]; do nvidia-smi --query-gpu=index,power.draw,utilization.gpu --format=csv,noheader >> $O/$arm.power; sleep 5; done" > /dev/null 2>&1 < /dev/null &
  /usr/bin/time -f "%e %U %S %M" -o "$O/$arm.time" \
    mpirun --allow-run-as-root -n 5 "$bin" --o "$O/${arm}_run" \
      --continue "$D/Prod/opt_x${k}.star" $COMMON $acc > "$O/$arm.log" 2>&1
  echo "rc=$? arm=$arm acc=$acc scale=$k" > "$O/$arm.rc"
  echo done > "$O/$arm.done"
  echo "$arm: $(tail -1 "$O/$arm.time")"
}

G=$R/build-gpu/bin/relion_refine_mpi
C=$R/build-cpu/bin/relion_refine_mpi

# 1 GPU across the whole ladder: the headline comparator, and the only arm that gets all four points
run g1_x1  "$G" "--gpu 0"   1
run c_x1   "$C" "--cpu"     1
run g1_x3  "$G" "--gpu 0"   3
run c_x3   "$C" "--cpu"     3
run g1_x8  "$G" "--gpu 0"   8
run g1_x25 "$G" "--gpu 0"   25
# 2 GPUs only at the top of the ladder: §7 already has the 1->2 slope at 4,452 (1.321x), and the
# question that is open is whether that slope improves once the job is big enough to fill them.
run g2_x25 "$G" "--gpu 0:1" 25
echo GPU_LADDER_DONE > "$O/campaign.done"
