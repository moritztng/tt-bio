#!/bin/bash
# The coarse leg's arms. One binary (build-fine, unchanged), and the variable is what the hooks do.
#
# Phase 0a, three walls that decompose the bridge, one iteration each (--iter 13, continuing from
# it012, the same starting point every arm in this lineage uses):
#
#   ref       PYTHONPATH unset, so TTBridge's import of tt_bio.cryoem.relion fails once and
#             g_usable stays false. RELION runs its own kernels with ZERO Python crossings. This is
#             the reference wall on this box, this binary and this box's current load -- not a
#             number carried over from another run.
#   nullbase  BACKEND=nullbase -> both hooks return False the instant they are entered. nullbase-ref
#             is the crossing alone: the GIL, the argument tuple, the memoryview construction.
#   null      BACKEND=null -> the coarse hook additionally does the buffer traffic the device path
#             will do (the model keyed rather than copied, everything else copied), then declines.
#             null-nullbase is the marshalling. P1 grades null/ref <= 1.15.
#
# --cpu is mandatory. Without it RELION never enters src/acc, the bridge is never called, and the
# run looks healthy and full-speed the whole way (relion-acc-backend §2.5). Every arm's counter line
# is harvested afterwards, and the ref arm is the one arm that is SUPPOSED to show no bridge line.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
WT=$S/tt-bio-coarse                # stable clone, deliberately outside any worktree
BIN=$S/relion/build-fine/bin/relion_refine_mpi
BL=$HOME/.coworker/scripts/benchlock.sh
mkdir -p "$S/coarse"

COMMON="--continue $T/Refine3D/job019/run_it012_optimiser.star \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"

inner() {
  arm=$1; bk=$2; extra=${3:-}
  [ -f "$S/build-fine.ok" ] || { echo "BUILD_MISSING" > "$S/coarse/$arm.rc"; return 1; }
  cd "$T" || return 1
  if [ "$arm" = "ref" ]; then
    pp=""                          # no PYTHONPATH: the bridge cannot load, RELION runs itself
  else
    pp="-x PYTHONPATH=$WT"
  fi
  /usr/bin/time -f "%e %U %S %M" -o "$S/coarse/$arm.time" \
    mpirun -n 5 $pp -x TT_RELION_BACKEND="$bk" -x TT_RELION_PROFILE=1 \
      "$BIN" --o "$S/coarse/${arm}_run" $COMMON $extra > "$S/coarse/$arm.log" 2>&1
  echo "rc=$? arm=$arm backend=$bk extra=$extra" > "$S/coarse/$arm.rc"
}

run_arm() {
  arm=$1; bk=$2; extra=${3:-}
  [ -f "$S/coarse/$arm.done" ] && return 0        # relaunch-safe: never repeat a finished arm
  rm -f "$S/coarse/$arm".* 2>/dev/null
  uptime > "$S/coarse/$arm.loadavg"
  BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-1800} "$BL" \
    worker:relion-kernel-coarse-projection -- bash "$0" inner "$arm" "$bk" "$extra"
  bl=$?
  uptime >> "$S/coarse/$arm.loadavg"
  if [ "$bl" = "75" ]; then                       # benchlock timed out: deliberately NOT measured
    echo "benchlock=75 arm=$arm NOT RUN" > "$S/coarse/$arm.rc"
    return 75
  fi
  echo DONE > "$S/coarse/$arm.done"
}

if [ "${1:-}" = "inner" ]; then inner "$2" "$3" "${4:-}"; exit $?; fi

case "${1:-p0a}" in
  ref)      run_arm ref      none     "--iter 13" ;;
  nullbase) run_arm nullbase nullbase "--iter 13" ;;
  null)     run_arm null     null     "--iter 13" ;;
  p0a)      run_arm ref none "--iter 13" \
              && run_arm nullbase nullbase "--iter 13" \
              && run_arm null null "--iter 13" ;;
esac
echo COARSE_CAMPAIGN_DONE > "$S/coarse/campaign.done"
