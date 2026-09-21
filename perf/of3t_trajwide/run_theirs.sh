#!/bin/bash
# The reference side of the wider rung, detached: upstream 0.4.3 float64 DiffusionModule,
# 20 steps, 4 accumulation samples of 12 noise levels each.
#   run 1  arm `theirs`      + the in-process A/A (two instances, same process, same drive)
#   run 2  arm `theirs_aa2`  the cross-process A/A
# CPU only, 12 threads. The card stays free for the device side.
#
# The chain records its OWN exit, per arm, in `chain_theirs.log`. The previous version wrote
# nothing about itself, so when the ours-side chain died between arms the only evidence was a
# log whose last line looked like progress (D140). stderr goes to its own file per arm so a
# silent death is distinguishable from a clean finish.
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/trajwide
mkdir -p "$L"
C="$L/chain_theirs.log"

say() { echo "$(date -u +%FT%TZ) $*" >> "$C"; }

say "chain start pid=$$ cwd=$PWD"
for spec in "theirs --aa-in-process" "theirs_aa2 "; do
    set -- $spec
    arm=$1; shift
    say "arm $arm start"
    $PY perf/of3t_trajwide/trajwide.py --side theirs --arm "$arm" --threads 12 "$@" \
        >> "$L/$arm.log" 2>> "$L/$arm.err"
    rc=$?
    echo "=== $arm done rc=$rc $(date -u +%FT%TZ)" >> "$L/$arm.log"
    say "arm $arm done rc=$rc steplog=$([ -s "$L/steplog_$arm.json" ] && echo yes || echo NO)"
    [ $rc -ne 0 ] && say "arm $arm FAILED, stderr tail: $(tail -3 "$L/$arm.err" | tr '\n' ' ')"
done
say "chain done rc=0"
