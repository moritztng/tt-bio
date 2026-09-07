#!/bin/sh
# The whole parity package for the two engine changes on this branch, on one chip at a time.
#
#   sh parity_run.sh <main-tree> <fix-tree>
#
# Legs, in this order:
#   aa_256   main, seed 42     -- the A/A control's first half
#   aa2_256  main, seed 42     -- and its second. These two MUST match, or nothing below means
#                                anything: a matching A/B would only show the fold is
#                                deterministic, and a differing one could be run-to-run noise.
#   a_128 a_256 a_512          -- main
#   b_128 b_256 b_512          -- main + the two fixes
#
# 256 is folded three times on main on purpose: aa/aa2 for the control and `a_256` reused from
# aa, so the A/B at 256 is scored against the same bits the control was.
set -u
MAIN=$1; FIX=$2
HERE=$(dirname "$0")
RUN=/home/cust-team/mthuening/ceilof3/rundir
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
LOG=$RUN/parity/parity.log
mkdir -p "$RUN/parity"

leg() {  # tag rung tree
  grep -q "^$1 " "$LOG" 2>/dev/null && { echo "$1 already run, skipping" >> "$LOG"; return; }
  sh "$HERE/parity_ab.sh" "$1" "$2" "$3" 42 >> "$LOG" 2>&1
}

leg aa_256  par_256 "$MAIN"
leg aa2_256 par_256 "$MAIN"
for n in 128 256 512; do leg "a_$n" "par_$n" "$MAIN"; done
for n in 128 256 512; do leg "b_$n" "par_$n" "$FIX"; done

{
  echo "=== A/A control (same tree, same seed, twice) -- must be identical ==="
  "$PY" "$HERE/parity_cmp.py" "$RUN/parity/aa_256" "$RUN/parity/aa2_256" --label "AA_256"
  echo "=== A/B (main vs main+fixes) ==="
  for n in 128 256 512; do
    "$PY" "$HERE/parity_cmp.py" "$RUN/parity/a_$n" "$RUN/parity/b_$n" --label "AB_$n"
  done
} >> "$LOG" 2>&1
echo "PARITY DONE $(date -u +%FT%TZ)" >> "$LOG"
