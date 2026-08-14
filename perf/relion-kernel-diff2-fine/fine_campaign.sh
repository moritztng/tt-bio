#!/bin/bash
# The fine pass, three arms. One binary (build-fine), and the variable is what the two hooks do.
#
#   shape  BACKEND=ttnn + TT_RELION_SHAPE -> both hooks decline, RELION runs its own kernels, and
#          every fine call logs its shape first. This is the shape characterisation across a WHOLE
#          refinement, which is the input to every parallelisation choice downstream. It is NOT a
#          timing arm: the decline path still crosses into Python once per call, so its wall is
#          above the reference's own 922.19 s and must not be quoted as a reference wall.
#   check  BACKEND=torch + TT_RELION_CHECK=1, one iteration -> the fine hook computes into a
#          private buffer and declines, RELION's own kernel writes diff2s, and TTBridge grades one
#          against the other. This is the kernel-level parity number, with RELION's own kernel on
#          RELION's own inputs as the oracle.
#   tt     BACKEND=torch, both hooks live, to convergence -> the refinement-level parity gate:
#          3.79378 A unmasked, 4.033896 A after relion_postprocess, and e2e_disagree against
#          ref_run_data.star.
#
# --cpu is mandatory. Without it RELION never enters src/acc, the bridge is never called, and the
# run looks healthy and full-speed the whole way (relion-acc-backend §2.5).
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
WT=$S/tt-bio-fine                 # stable clone, deliberately outside any worktree
BIN=$S/relion/build-fine/bin/relion_refine_mpi
BL=$HOME/.coworker/scripts/benchlock.sh
mkdir -p "$S/fine"

COMMON="--continue $T/Refine3D/job019/run_it012_optimiser.star \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"

# The build runs INSIDE the lock: a 28-way make on qb1 is exactly the co-tenant noise benchlock
# exists to keep off someone else's timed arm.
inner() {
  arm=$1; bk=$2; extra=${3:-}
  if [ ! -f "$S/build-fine.ok" ]; then
    mkdir -p "$S/relion/build-fine" && cd "$S/relion/build-fine" || return 1
    cmake .. -DALTCPU=ON -DTT=ON -DCUDA=OFF -DGUI=OFF -DMKLFFT=OFF -DFETCH_WEIGHTS=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_FLAGS="-DTIMING" \
      -DCMAKE_INSTALL_PREFIX="$S/relion/build-fine" \
      > "$S/cmakeFINE.log" 2>&1 \
      && make -j 28 > "$S/makeFINE.log" 2>&1 \
      && touch "$S/build-fine.ok"
  fi
  [ -f "$S/build-fine.ok" ] || { echo "BUILD_FAIL" > "$S/fine/$arm.rc"; return 1; }
  cd "$T" || return 1
  env_shape=""; env_check=""
  [ "$arm" = "shape" ] && env_shape="-x TT_RELION_SHAPE=$S/fine/shape"
  [ "$arm" = "check" ] && env_check="-x TT_RELION_CHECK=1"
  /usr/bin/time -f "%e %U %S %M" -o "$S/fine/$arm.time" \
    mpirun -n 5 -x PYTHONPATH="$WT" -x TT_RELION_BACKEND="$bk" -x TT_RELION_PROFILE=1 \
      $env_shape $env_check \
      "$BIN" --o "$S/fine/${arm}_run" $COMMON $extra > "$S/fine/$arm.log" 2>&1
  echo "rc=$? arm=$arm backend=$bk extra=$extra" > "$S/fine/$arm.rc"
}

run_arm() {
  arm=$1; bk=$2; extra=${3:-}
  [ -f "$S/fine/$arm.done" ] && return 0        # relaunch-safe: never repeat a finished arm
  rm -f "$S/fine/$arm".* 2>/dev/null
  uptime > "$S/fine/$arm.loadavg"
  BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-3600} "$BL" worker:relion-kernel-diff2-fine -- \
    bash "$0" inner "$arm" "$bk" "$extra"
  bl=$?
  uptime >> "$S/fine/$arm.loadavg"
  # benchlock exit 75 means it timed out waiting for the box and deliberately did NOT measure.
  if [ "$bl" = "75" ]; then
    echo "benchlock=75 arm=$arm NOT RUN" > "$S/fine/$arm.rc"
    return 75
  fi
  echo DONE > "$S/fine/$arm.done"
}

if [ "${1:-}" = "inner" ]; then inner "$2" "$3" "${4:-}"; exit $?; fi

case "${1:-all}" in
  shape) run_arm shape ttnn ;;
  check) run_arm check torch "--iter 13" ;;
  tt)    run_arm tt    torch ;;
  all)   run_arm shape ttnn && run_arm check torch "--iter 13" && run_arm tt torch ;;
esac
echo FINE_CAMPAIGN_DONE > "$S/fine/campaign.done"
