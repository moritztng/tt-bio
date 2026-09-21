#!/bin/bash
# The reference side of the wider rung, detached: upstream 0.4.3 float64 DiffusionModule,
# 20 steps, 4 accumulation samples of 12 noise levels each.
#   run 1  arm `theirs`      + the in-process A/A (two instances, same process, same drive)
#   run 2  arm `theirs_aa2`  the cross-process A/A
# CPU only. The card stays free for the device side.
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/trajwide
mkdir -p "$L"
$PY perf/of3t_trajwide/trajwide.py --side theirs --arm theirs --threads 12 \
    --aa-in-process >> "$L/theirs.log" 2>&1
echo "=== theirs done rc=$? $(date -u +%FT%TZ)" >> "$L/theirs.log"
$PY perf/of3t_trajwide/trajwide.py --side theirs --arm theirs_aa2 --threads 12 \
    >> "$L/theirs_aa2.log" 2>&1
echo "=== theirs_aa2 done rc=$? $(date -u +%FT%TZ)" >> "$L/theirs_aa2.log"
