#!/bin/bash
# main1536_seeds.sh <main tree> <seed>...: on main, one fresh eal_1536 fold per PYTHONHASHSEED.
# hashseed.py sorts main's hash seeds into template assignments (0-3,5,6 / 4 / 7 over seeds 0-7);
# folding one seed per assignment gives main's structural spread at 1536 without drawing folds
# until every assignment turns up.
set -u
M=$1; shift
for s in "$@"; do
  PYTHONHASHSEED=$s "$M/perf/mgx_det/run.sh" fresh 1 perf/mgx_combos/inputs/eal_1536.yaml "main1536_h$s"
done
