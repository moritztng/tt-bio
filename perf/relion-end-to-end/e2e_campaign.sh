#!/bin/bash
# E1 + E2: two complete RELION refinements to convergence, one binary, one variable.
#
# The variable is TT_RELION_BACKEND. `ttnn` makes the bridge decline every coarse call, so RELION
# runs its own CpuKernels; `torch` makes the bridge handle it in tt_bio/cryoem/relion.py. Both arms
# use the SAME binary (build-e2e: TT=ON, and the by-stage instrument compiled in), so the A/B has
# exactly one variable. build-timing has the instrument but no bridge and build-tt has the bridge but
# predates the instrument, which is why neither is used here.
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

# --- stage 0: the one binary
if [ ! -f "$S/build-e2e.ok" ]; then
  mkdir -p "$S/relion/build-e2e" && cd "$S/relion/build-e2e" || exit 1
  cmake .. -DALTCPU=ON -DTT=ON -DCUDA=OFF -DGUI=OFF -DMKLFFT=OFF -DFETCH_WEIGHTS=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$S/relion/build-e2e" \
    > "$S/cmakeE2E.log" 2>&1 \
    && make -j 28 > "$S/makeE2E.log" 2>&1 \
    && touch "$S/build-e2e.ok"
fi
if [ ! -f "$S/build-e2e.ok" ]; then echo BUILD_FAIL > "$S/e2e/campaign.done"; exit 1; fi

cd "$T" || exit 1
COMMON="--continue $T/Refine3D/job019/run_it012_optimiser.star \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"

run_arm() {
  arm=$1; bk=$2
  [ -f "$S/e2e/$arm.done" ] && return 0        # relaunch-safe: never repeat a finished arm
  uptime > "$S/e2e/$arm.loadavg"
  BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=3600 "$BL" worker:relion-end-to-end -- \
    /usr/bin/time -f "%e %U %S %M" -o "$S/e2e/$arm.time" \
      mpirun -n 5 -x PYTHONPATH="$WT" -x TT_RELION_BACKEND="$bk" -x TT_RELION_PROFILE=1 \
        "$BIN" --o "$S/e2e/${arm}_run" $COMMON > "$S/e2e/$arm.log" 2>&1
  echo "rc=$? arm=$arm backend=$bk" > "$S/e2e/$arm.rc"
  uptime >> "$S/e2e/$arm.loadavg"
  echo DONE > "$S/e2e/$arm.done"
}

run_arm ref ttnn      # E1: RELION's own kernels, its own answer, its own wall
run_arm tt   torch    # E2: the coarse pass in our code, every iteration
echo CAMPAIGN_DONE > "$S/e2e/campaign.done"
