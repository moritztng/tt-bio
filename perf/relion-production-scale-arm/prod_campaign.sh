#!/bin/bash
# The production-scale arm: one E-step iteration at four particle counts, both backends.
#
# `relion-end-to-end` §9 lost to a single H200 on a 4,452-particle job and named the reason: the
# job sits below `relion-intercard-scaling` §15.3's measured ~7,060-particle crossover. Deciding
# whether that verdict survives at production scale needs the scaling LAW, not one more point, so
# this runs the same iteration at 1x / 3x / 8x / 25x (4,452 -> 111,300 particles) and fits it.
#
# One iteration, not a whole refinement, and that is deliberate. `--continue` from RELION's own
# `run_it012_optimiser.star` with `--auto_iter_max 13` pins the trajectory: same sampling order,
# same current image size, same reference maps at every scale. A refinement to convergence would
# take a different number of iterations at each scale (more particles -> different convergence),
# so its wall would confound "more particles" with "more iterations" and no slope could be read
# out of it. The whole-refinement composition is done in the doc, from this slope plus the
# per-iteration counts `relion-end-to-end` §4 already measured.
#
# Same binary as the e2e campaign (`build-e2e`: TT=ON plus -DTIMING) and same single variable
# (`TT_RELION_BACKEND`), so every number here composes with §4's stage split directly.
set -u
S=/home/ttuser/relion-scratch
T=$S/Tutorial5.0
WT=$S/tt-bio                 # stable clone, deliberately outside any worktree
BIN=$S/relion/build-e2e/bin/relion_refine_mpi
BL=$HOME/.coworker/scripts/benchlock.sh
O=$S/prod
mkdir -p "$O"

COMMON="--auto_iter_max 13 --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images"

inner() {
  arm=$1; bk=$2; k=$3
  cd "$T" || return 1
  /usr/bin/time -f "%e %U %S %M" -o "$O/$arm.time" \
    mpirun -n 5 -x PYTHONPATH="$WT" -x TT_RELION_BACKEND="$bk" -x TT_RELION_PROFILE=1 \
      "$BIN" --o "$O/${arm}_run" --continue "$T/Prod/opt_x${k}.star" $COMMON \
      > "$O/$arm.log" 2>&1
  echo "rc=$? arm=$arm backend=$bk scale=$k" > "$O/$arm.rc"
}

run_arm() {
  arm=$1; bk=$2; k=$3
  [ -f "$O/$arm.done" ] && { echo "skip $arm (done)"; return 0; }
  uptime > "$O/$arm.loadavg"
  BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=${BLWAIT:-14400} "$BL" \
    worker:relion-production-scale-arm -- bash "$0" inner "$arm" "$bk" "$k"
  bl=$?
  uptime >> "$O/$arm.loadavg"
  # benchlock 75 means it gave up waiting for a quiet box and deliberately did NOT measure. Never
  # write a done-marker for an arm that never ran -- a relaunch would then skip it silently.
  if [ "$bl" = "75" ]; then
    echo "benchlock=75 arm=$arm NOT RUN" > "$O/$arm.rc"
    echo "BENCHLOCK_TIMEOUT $arm"
    return 75
  fi
  echo DONE > "$O/$arm.done"
  echo "finished $arm: $(cat "$O/$arm.time" 2>/dev/null | tail -1)"
}

if [ "${1:-}" = "inner" ]; then inner "$2" "$3" "$4"; exit $?; fi

# Order matters: the reference ladder first and cheapest-first, so a truncated campaign still
# yields a fit. The bridge arm is 4.3x slower (§2) and only needs enough points to show that the
# parity result holds at scale -- it is not, and must never be quoted as, the port's performance.
SPECS=${SPECS:-"ref1:ttnn:1 ref3:ttnn:3 ref8:ttnn:8 ref25:ttnn:25"}
for spec in $SPECS; do
  IFS=: read -r arm bk k <<< "$spec"
  run_arm "$arm" "$bk" "$k" || exit $?
done
echo CAMPAIGN_DONE > "$O/campaign.done"
