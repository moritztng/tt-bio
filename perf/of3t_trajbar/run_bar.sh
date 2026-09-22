#!/bin/bash
# The bar's two arms, detached, CPU only, no card.
#   bf16mixed      upstream 0.4.3 bf16-mixed over an fp32 master, 20 steps, the treatment
#   bf16mixed_aa2  the same arm in a SECOND process, 3 rungs, the arm's own determinism check
# 10 threads, which is what the float64 reference arm this is compared against ran at
# (`perf/of3t_trajwide/run_theirs.sh`), so the wall clocks are comparable.
#
# Each arm records its own rc in its own marker. An arm with a log and no marker is a FAILED
# arm, not a finished one (of3t-trajwide D140).
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python
R=/home/ttuser/of3t_runs/trajbar
mkdir -p "$R"
C="$R/chain.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$C"; }

say "chain start pid=$$ cwd=$PWD loadavg=$(cat /proc/loadavg)"
for spec in "bf16mixed 20" "bf16mixed_aa2 3"; do
    set -- $spec
    arm=$1; steps=$2
    say "arm $arm start steps=$steps"
    $PY perf/of3t_trajbar/trajbar.py --run --mode bf16mixed --arm "$arm" \
        --steps "$steps" --threads 10 >> "$R/$arm.log" 2>> "$R/$arm.err"
    rc=$?
    echo "=== $arm done rc=$rc $(date -u +%FT%TZ)" >> "$R/$arm.log"
    echo "arm=$arm rc=$rc mode=bf16mixed steps=$steps at=$(date -u +%FT%TZ)" > "$R/$arm.done"
    say "arm $arm done rc=$rc steplog=$([ -s "$R/steplog_$arm.json" ] && echo yes || echo NO)"
    [ $rc -ne 0 ] && say "arm $arm FAILED, stderr tail: $(tail -3 "$R/$arm.err" | tr '\n' ' ')"
done
say "chain done rc=0 loadavg=$(cat /proc/loadavg)"
