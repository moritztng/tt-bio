#!/bin/bash
# Block on every qb2 chip lock; the first one granted claims the leg (mkdir), the rest release.
S=/home/ttuser/.coworker/wt/rfd4-cells; R=${1:?outdir}
for c in 0 2 1 3; do
  flock -w 10800 $HOME/spd_qb2_chip$c.lock bash -c "mkdir $R/claim 2>/dev/null && LEGS=\"${LEGS:-}\" exec bash $S/perf/rfd4/e2e/cells.sh $c $R" &
done
wait
