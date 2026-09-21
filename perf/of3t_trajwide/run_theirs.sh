#!/bin/bash
# The reference side of the wider rung, detached: upstream 0.4.3 float64 DiffusionModule,
# 20 steps, 4 accumulation samples of 12 noise levels each.
#   run 1  arm `theirs`      + the in-process A/A (two instances, same process, same drive)
#   run 2  arm `theirs_aa2`  the cross-process A/A
# CPU only, 10 threads, which leaves the two device chains their 3 each on a 16-core box.
#
# The chain records its OWN exit, per arm, and drops a marker under perf/of3t_trajwide/runs/.
# An arm with a log and no marker is a FAILED arm, not a finished one (D140).
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python
R=/home/ttuser/of3t_runs/trajwide
G=perf/of3t_trajwide/runs
mkdir -p "$R" "$G"
C="$R/chain_theirs.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$C"; }

say "chain start pid=$$ cwd=$PWD"
for spec in "theirs --aa-in-process" "theirs_aa2 "; do
    set -- $spec
    arm=$1; shift
    say "arm $arm start"
    $PY perf/of3t_trajwide/trajwide.py --side theirs --arm "$arm" --threads 10 "$@" \
        >> "$R/$arm.log" 2>> "$R/$arm.err"
    rc=$?
    echo "=== $arm done rc=$rc $(date -u +%FT%TZ)" >> "$R/$arm.log"
    echo "arm=$arm rc=$rc side=theirs at=$(date -u +%FT%TZ)" > "$G/$arm.done"
    say "arm $arm done rc=$rc steplog=$([ -s "$R/steplog_$arm.json" ] && echo yes || echo NO)"
    [ $rc -ne 0 ] && say "arm $arm FAILED, stderr tail: $(tail -3 "$R/$arm.err" | tr '\n' ' ')"
done
say "chain done rc=0"
