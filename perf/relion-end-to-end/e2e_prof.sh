#!/bin/bash
# One RELION iteration on RELION's OWN kernels, with the by-stage instrument on, so the coarse share
# of the E-step is measured rather than derived from sampling counts. 5 MPI x 6 threads = 30 threads,
# matching P1's split in relion-acc-backend.md section 3.1 so the shares divide that table.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
BIN=$S/relion/build-timing/bin/relion_refine_mpi
mkdir -p "$S/e2e"
rm -f "$S/e2e/prof.rc" "$S/e2e/prof.done"
cd "$T" || exit 1
uptime > "$S/e2e/prof.loadavg"
/usr/bin/time -f "%e %M" -o "$S/e2e/prof.time" \
  mpirun -n 5 -x TT_RELION_PROFILE=1 \
    "$BIN" --o "$S/e2e/prof_run" \
    --continue "$T/Refine3D/job019/run_it012_optimiser.star" --auto_iter_max 13 \
    --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images \
    > "$S/e2e/prof.log" 2>&1
echo "rc=$?" > "$S/e2e/prof.rc"
uptime >> "$S/e2e/prof.loadavg"
echo DONE > "$S/e2e/prof.done"
