#!/bin/bash
# E1 + E2: two complete RELION refinements to convergence, one binary, one variable.
#
# The variable is TT_RELION_BACKEND. `ttnn` makes the bridge decline every coarse call, so RELION
# runs its own CpuKernels; `torch` makes the bridge handle it in tt_bio/cryoem/relion.py. Both arms
# use the SAME binary, so the A/B has exactly one variable.
#
# That binary is `build-e2e`: TT=ON **and** -DTIMING. It has to be both. TT=ON gets the bridge and
# the CTIC/CTOC instrument (the acc-internal regions -- oneParticle, the coarse and fine passes,
# storeWeightedSums); -DTIMING gets RELION's own Timer table (expectation_1/2/6, maximization,
# writeOutput, flatten solvent), which is where every row of the pipeline floor comes from. The
# first cut of this script built without -DTIMING and would have produced two complete refinements
# with no M-step or setup timing in either. Cost of the pair, measured in §0: 0.7%.
#
# Convergence for this job is iteration 16, not 17: RELION's own precalculated Refine3D/job019 runs
# `Expectation iteration 16` and then prints "Refinement has converged, stopping now" with a final
# unmasked resolution of 3.79378 A. Continuing from it012 is therefore 4 iterations, not 5.
#
# --cpu is mandatory. Without it RELION never enters src/acc and the bridge is never called, and the
# run looks healthy and full-speed the whole way.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
WT=$S/tt-bio                 # stable clone, deliberately outside any worktree
BIN=$S/relion/build-e2e/bin/relion_refine_mpi
BL=$HOME/.coworker/scripts/benchlock.sh
mkdir -p "$S/e2e"
rm -f "$S/e2e/campaign.done"

COMMON="--continue $T/Refine3D/job019/run_it012_optimiser.star \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"

# The build runs INSIDE the lock, not before it. A 28-way make on qb1 is exactly the co-tenant noise
# benchlock exists to keep off someone else's timed arm, and this box currently has another worker's
# perf regression holding it.
inner() {
  arm=$1; bk=$2
  if [ ! -f "$S/build-e2e.ok" ]; then
    mkdir -p "$S/relion/build-e2e" && cd "$S/relion/build-e2e" || return 1
    cmake .. -DALTCPU=ON -DTT=ON -DCUDA=OFF -DGUI=OFF -DMKLFFT=OFF -DFETCH_WEIGHTS=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_FLAGS="-DTIMING" \
      -DCMAKE_INSTALL_PREFIX="$S/relion/build-e2e" \
      > "$S/cmakeE2E.log" 2>&1 \
      && make -j 28 > "$S/makeE2E.log" 2>&1 \
      && touch "$S/build-e2e.ok"
  fi
  [ -f "$S/build-e2e.ok" ] || { echo "BUILD_FAIL" > "$S/e2e/$arm.rc"; return 1; }
  cd "$T" || return 1
  /usr/bin/time -f "%e %U %S %M" -o "$S/e2e/$arm.time" \
    mpirun -n 5 -x PYTHONPATH="$WT" -x TT_RELION_BACKEND="$bk" -x TT_RELION_PROFILE=1 \
      "$BIN" --o "$S/e2e/${arm}_run" $COMMON > "$S/e2e/$arm.log" 2>&1
  echo "rc=$? arm=$arm backend=$bk" > "$S/e2e/$arm.rc"
}

run_arm() {
  arm=$1; bk=$2
  [ -f "$S/e2e/$arm.done" ] && return 0        # relaunch-safe: never repeat a finished arm
  uptime > "$S/e2e/$arm.loadavg"
  BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=3600 "$BL" worker:relion-end-to-end -- \
    bash "$0" inner "$arm" "$bk"
  bl=$?
  uptime >> "$S/e2e/$arm.loadavg"
  # benchlock exit 75 means it timed out waiting for the box and deliberately did NOT measure. The
  # first cut wrote the done-marker unconditionally, so a relaunch would have skipped an arm that
  # never ran. Mark done only when the arm actually happened.
  if [ "$bl" = "75" ]; then
    echo "benchlock=75 arm=$arm NOT RUN" > "$S/e2e/$arm.rc"
    return 75
  fi
  echo DONE > "$S/e2e/$arm.done"
}

if [ "${1:-}" = "inner" ]; then inner "$2" "$3"; exit $?; fi

run_arm ref ttnn      # E1: RELION's own kernels, its own answer, its own wall
run_arm tt   torch    # E2: the coarse pass in our code, every iteration
echo CAMPAIGN_DONE > "$S/e2e/campaign.done"
