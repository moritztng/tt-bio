#!/usr/bin/env bash
# The four cells of the 256-rung mode discriminator, in one benchlock hold.
#
#   quiet  256   H2's arm and the quiet reference
#   quiet  512   H2's required negative control
#   loaded 256   H1's arm
#   loaded 512   H1's control: does host load move a rung that is not 256
#
# One lock acquisition for all four so no other worker's timed run lands between the arms an
# A/B is being read across. Invoke through benchlock, not directly.
set -u
D=/home/ttuser/.coworker/wt/c14-gate-256-warmup/perf/c14_gate_256
LOADN=${LOADN:-24}
NP=${NP:-3}
for spec in "q256 256 $NP 0" "q512 512 $NP 0" "l256 256 $NP $LOADN" "l512 512 $NP $LOADN"; do
  # shellcheck disable=SC2086
  bash "$D/run_cell.sh" $spec
  echo "---- cell done rc=$? ----"
done
