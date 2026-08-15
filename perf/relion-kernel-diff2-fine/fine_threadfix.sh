#!/bin/bash
# One iteration of the `tt` arm with torch's intra-op thread count pinned.
#
# The campaign left TT_RELION_TORCH_THREADS unset, so every one of the 5 MPI ranks asked torch for
# one thread per core on a 32-core box: 160 torch threads for 32 cores, inside a region the GIL
# already serialises. The 5-concurrent-rank proxy prices that at 34x on the coarse call. This runs
# it13 for real with the variable set, so the fix is measured in the refinement rather than beside
# it (tt-bio-isolated-op-timing-oversync-inflates-cost).
#
# Compare against the `tt` arm's own it13: 01:53:56 -> 02:14:30, 1234 s.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
WT=$S/tt-bio-fine
BIN=$S/relion/build-fine/bin/relion_refine_mpi
THR=${THR:-6}
ARM=${ARM:-thr}

cd "$T" || exit 1
rm -f "$S/fine/${ARM}_run"* 2>/dev/null
/usr/bin/time -f "%e %U %S %M" -o "$S/fine/$ARM.time" \
  mpirun -n 5 -x PYTHONPATH="$WT" -x TT_RELION_BACKEND=torch -x TT_RELION_PROFILE=1 \
    -x TT_RELION_TORCH_THREADS="$THR" \
    "$BIN" --o "$S/fine/${ARM}_run" \
    --continue "$T/Refine3D/job019/run_it012_optimiser.star" \
    --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images --iter 13 \
    > "$S/fine/$ARM.log" 2>&1
echo "rc=$? arm=$ARM threads=$THR" > "$S/fine/$ARM.rc"
cat "$S/fine/$ARM.rc"; cat "$S/fine/$ARM.time"
