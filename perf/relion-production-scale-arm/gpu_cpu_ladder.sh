#!/bin/bash
# The rented box's CPU ladder at the top two scales.
#
# The host correction (`relion-end-to-end` §7.3: qb1's EPYC is 1.749x slower than this class of Xeon)
# is what makes a Tenstorrent wall composed against qb1 comparable to a GPU wall measured here. That
# correction was a single ratio taken at 4,452 particles. The GPU ladder has just shown its own
# ratios move with particle count, so the correction has to be measured at production scale too.
set -u
O=/root/prod
D=/root/Tutorial5.0
C=/root/relion/build-cpu/bin/relion_refine_mpi
cd "$D" || exit 1
for k in 8 25; do
  arm=c_x$k
  [ -f "$O/$arm.done" ] && { echo "skip $arm"; continue; }
  rm -f "$O/$arm.rc"
  /usr/bin/time -f "%e %U %S %M" -o "$O/$arm.time" \
    mpirun --allow-run-as-root -n 5 "$C" --o "$O/${arm}_run" \
      --continue "$D/Prod/opt_x${k}.star" --auto_iter_max 13 --j 6 --pool 6 \
      --dont_combine_weights_via_disc --preread_images --cpu > "$O/$arm.log" 2>&1
  echo "rc=$? arm=$arm scale=$k" > "$O/$arm.rc"
  echo done > "$O/$arm.done"
  echo "$arm: $(tail -1 "$O/$arm.time")"
done
echo CPU_LADDER_DONE > "$O/campaign.done"
