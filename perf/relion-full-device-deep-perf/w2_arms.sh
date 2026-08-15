#!/bin/bash
# W2/P4 -- the paired refinements. One binary, one source tree, one link; the only variable is
# TT_COARSE_E, so the compiler is out of the comparison entirely.
#
#   e1   TT_COARSE_E=1   control. rest = orientation_num % 1 = 0, so the whole orientation range
#                        goes through the blocked kernel at eulers_per_block=1 -- which is exactly
#                        the remainder path RELION runs today, one orientation at a time.
#   e16  TT_COARSE_E=16  the fix: rest taken modulo the ORIENTATION block size instead of the PIXEL
#                        block size, so the blocked kernel finally engages.
#
# Both arms are full it13-17 runs (--iter 13 does not stop the run at 13; leg 3, coarse §8.2).
# Relaunch-safe: a finished arm is never repeated.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
BIN=$S/relion/build-e2e/bin/relion_refine_mpi
BL=$HOME/.coworker/scripts/benchlock.sh
O=$S/w2/arms
mkdir -p "$O"

COMMON="--continue $T/Refine3D/job019/run_it012_optimiser.star \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"

inner() {
  arm=$1; e=$2
  cd "$T" || return 1
  /usr/bin/time -f "%e %U %S %M" -o "$O/$arm.time" \
    mpirun -n 5 -x TT_COARSE_E="$e" "$BIN" --o "$O/${arm}_run" $COMMON --iter 13 \
      > "$O/$arm.log" 2>&1
  echo "rc=$? arm=$arm TT_COARSE_E=$e" > "$O/$arm.rc"
}

run_arm() {
  arm=$1; e=$2
  [ -f "$O/$arm.done" ] && return 0
  rm -f "$O/$arm".log "$O/$arm".rc "$O/$arm".time 2>/dev/null
  uptime > "$O/$arm.loadavg"
  BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-2400} "$BL" \
    worker:relion-full-device-deep-perf -- bash "$0" inner "$arm" "$e"
  bl=$?
  uptime >> "$O/$arm.loadavg"
  if [ "$bl" = "75" ]; then
    echo "benchlock=75 arm=$arm NOT RUN" > "$O/$arm.rc"
    return 75
  fi
  echo DONE > "$O/$arm.done"
}

if [ "${1:-}" = "inner" ]; then inner "$2" "$3"; exit $?; fi

case "${1:-both}" in
  e1)   run_arm e1  1 ;;
  e16)  run_arm e16 16 ;;
  both) run_arm e1 1; run_arm e16 16 ;;
esac
echo W2_ARMS_DONE > "$O/campaign.done"
