#!/bin/bash
# The two arms the harvested campaign left owed, chained under one benchlock hold.
#
#   bash /home/ttuser/relion-scratch/e2e_ctrl.sh          # detached, relaunch-safe per arm
#
# ARM `ref2` -- the control the parity verdict needs. The harvest found the tt arm's assignments
# drifting from the ref arm's over the trajectory: 0, 0, 6, 13, 18 particles of 4452 reassigned at
# it012..it016. Read alone that is "the bridge changes RELION's answer, and it compounds". But
# relion-acc-backend §4.8 measured that RELION's own ALTCPU reconstruction is NOT bit-reproducible
# run to run (same binary, same input, different half-map sha256), and a refinement feeds its own
# output back in. So a second REFERENCE arm, ref against ref, is the only thing that separates a
# bridge effect from RELION's own nondeterminism amplified by feedback. Identical to the ref arm in
# every respect including TT_RELION_BACKEND=ttnn; the only difference is that it is a second run.
#
# PRE-REGISTERED, written before the arm ran: if the drift is RELION's own, ref2-vs-ref shows a
# reassignment trajectory of the same shape and order as tt-vs-ref (single digits growing to tens by
# it016), and the bridge is exonerated. If ref2-vs-ref is 0/4452 at every iteration, the 18 particles
# ARE the bridge and the parity claim has to say so. Anything in between is reported as in between.
#
# ARM `tt1t` -- the one named lever on the "today" column, §7 item 5b. The bridge arm came in at
# 3988.48 s against the reference arm's 922.19 s, and its per-particle cost got WORSE with more MPI
# ranks: §4.5 measured 325 s per follower for 2226 particles on -n 3 (0.146 s/particle) against this
# campaign's 636.334 s for 1113 on -n 5 (0.572 s/particle), same box, same binary. Candidate
# mechanism: torch sizes its intra-op pool per process from the core count, so four ranks ask the
# 32-core box for twice the threads two ranks did, on top of RELION's own --j 6 per rank.
# TT_RELION_TORCH_THREADS=1 is the one-line arm. One iteration only: the control is the SAME
# binary's measured it013 expectation of 636.334 s from the campaign, so this needs no second arm.
#
# PRE-REGISTERED: if oversubscription is the mechanism, it013 `expectation` lands below 424 s
# (1.5x). Below 1.1x means the mechanism is NOT thread oversubscription and the 5b hypothesis is
# refuted rather than confirmed -- which is a result, and it must be written as one.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
WT=$S/tt-bio
BIN=$S/relion/build-e2e/bin/relion_refine_mpi
BL=$HOME/.coworker/scripts/benchlock.sh
mkdir -p "$S/e2e"

COMMON="--continue $T/Refine3D/job019/run_it012_optimiser.star \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"

inner() {
  arm=$1; bk=$2; thr=$3; extra=$4
  cd "$T" || return 1
  /usr/bin/time -f "%e %U %S %M" -o "$S/e2e/$arm.time" \
    mpirun -n 5 -x PYTHONPATH="$WT" -x TT_RELION_BACKEND="$bk" -x TT_RELION_PROFILE=1 \
      -x TT_RELION_TORCH_THREADS="$thr" \
      "$BIN" --o "$S/e2e/${arm}_run" $COMMON $extra > "$S/e2e/$arm.log" 2>&1
  echo "rc=$? arm=$arm backend=$bk torch_threads=$thr" > "$S/e2e/$arm.rc"
}

run_arm() {
  arm=$1; bk=$2; thr=$3; extra=$4
  [ -f "$S/e2e/$arm.done" ] && return 0
  uptime > "$S/e2e/$arm.loadavg"
  BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=3600 "$BL" worker:relion-end-to-end -- \
    bash "$0" inner "$arm" "$bk" "$thr" "$extra"
  bl=$?
  uptime >> "$S/e2e/$arm.loadavg"
  # benchlock 75 is "timed out waiting for the box and deliberately did not measure". Marking that
  # done would make a relaunch skip an arm that never ran.
  if [ "$bl" = "75" ]; then
    echo "benchlock=75 arm=$arm NOT RUN" > "$S/e2e/$arm.rc"
    return 75
  fi
  echo done > "$S/e2e/$arm.done"
}

if [ "${1:-}" = "inner" ]; then inner "$2" "$3" "$4" "$5"; exit $?; fi

[ -x "$BIN" ] || { echo "no binary at $BIN"; exit 1; }
run_arm ref2 ttnn  0 ""
run_arm tt1t torch 1 "--auto_iter_max 13"
echo CTRL_DONE > "$S/e2e/ctrl.done"
