#!/usr/bin/env bash
# Pass 2: price the three candidate gate policies directly, at n=10.
#
# Pass 1 ran 3 processes x 5 folds and found the first fold of a fresh process is the slow one.
# The gate spends one fresh process per fold, so every cell it records is a fold-0 draw. What
# pass 1 could not settle at n=3 is whether ONE in-process warm-up fold is enough, because two
# of its three 256 processes were still decaying at fold 3.
#
#   w256  10 processes x 4 folds at 256   fold0 = today, fold1 = warm-up policy, med(1..3) = +median
#   w512   6 processes x 3 folds at 512   the same at the rung the exponent divides by
#
# Both quiet, one benchlock hold, clock forced to 1350 and sampled during, load sampled at 1 Hz.
set -u
D=/home/ttuser/.coworker/wt/c14-gate-256-warmup/perf/c14_gate_256
FOLDS=4 bash "$D/run_cell.sh" w256 256 10 0
echo "---- cell done rc=$? ----"
FOLDS=3 bash "$D/run_cell.sh" w512 512 6 0
echo "---- cell done rc=$? ----"
