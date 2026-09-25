#!/bin/bash
# The detached chain this row leaves on card 1: the interleaved A/B, then the float64 grade.
# Rooted in this worktree deliberately (fleet hygiene defers removal while a live process is
# under the path); both legs write under perf/bcx_tmplseam/runs/.
set -u
WT=/home/ttuser/.coworker/wt/bcx-tmplseam
cd "$WT" || exit 1
perf/bcx_tmplseam/launch.sh round_ab perf/bcx_tmplseam/round_ab.py --rounds 12   > perf/bcx_tmplseam/runs/round_ab.log 2>&1
echo "round_ab exit $?" >> perf/bcx_tmplseam/runs/round_ab.log
perf/bcx_tmplseam/launch.sh grade perf/bcx_tmplseam/grade.py --envelope   > perf/bcx_tmplseam/runs/grade.log 2>&1
echo "grade exit $?" >> perf/bcx_tmplseam/runs/grade.log
