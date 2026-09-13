#!/usr/bin/env bash
# The wheel-vs-source A/B. Same worktree, same fixture, same protocol, the ONLY variable is which
# tt-metal the process imports: the pip ttnn 0.68.0 wheel out of the venv, or the source build at
# that wheel own tag (v0.68.0 / 1452925b) in /home/ttuser/tt-metal-b2z.
#
# Why a driver and not --ab-env: the two arms are different native libraries, so an arm is a
# process, not a flag. Interleaving is therefore at process granularity and the order alternates
# so a monotone drift across the run cannot masquerade as an effect. Each process runs its own
# cold fold (discarded) and its own plain/instr reps, so each arm carries its own A/A floor.
set -u
ROUNDS="${1:-3}"
REPS="${2:-2}"
. /home/ttuser/scratch/uws_env.sh
cd "$WT"
run_one() {  # $1 arm  $2 tag  $3 phases
  ( set -e
    if [ "$1" = wheel ]; then arm_wheel; else arm_source; fi
    echo "=== $(date -u +%H:%M:%S) arm=$1 tag=$2 phases=$3 load=$(cut -d" " -f1 /proc/loadavg)"
    $PY perf/b2x-baseline-attrib/baseline_attrib.py \
        --phases "$3" --reps "$REPS" --size 512 \
        --out  "$ART/$2.json" --cifdir "$ART/cif_$2" ) 2>&1 | grep -Ev "^\s*$" | \
    grep -E "===|plain|instr|baseline\]|control\]|cold|Error|error|FATAL|Traceback" | head -40
}
for r in $(seq 0 $((ROUNDS-1))); do
  if [ $((r % 2)) -eq 0 ]; then ORDER="wheel source"; else ORDER="source wheel"; fi
  for a in $ORDER; do run_one "$a" "${a}_r${r}" baseline; done
done
