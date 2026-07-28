#!/bin/bash
# p27: price the p26 device gather at a real 200-step design, A/B'd against its parent.
#
# The parent tree is a `git archive` of f633db829 (the commit before the gather), the
# same rig run_p10_sweep.sh uses, so an interrupted run never leaves a dirty tree and
# the two trees differ only by the code under test -- the harness is copied, not forked.
#
# Two repeats per cell with the tree order reversed on the second, because alternating
# order alone does not cancel thermal bias in this port (a warmer card reads slower;
# see rfd3-p14). 20 timed steps plus the one-time table build, which p27's harness puts
# inside the timed region, compose into the 200-step number: build + 199 * steady.
#
#   scripts/rfd3_port/run_p27_sweep.sh ab   <fixture> [<fixture> ...]
#   scripts/rfd3_port/run_p27_sweep.sh full <fixture> [<fixture> ...]
# fixtures: iai40 iai80 iai150 mpro iai250
set -u
WT=/home/moritz/.coworker/wt/tt-bio-rfdiffusion3-largedesign-gap-p27
REF=/tmp/p27_ref
PY=/home/moritz/tt-bio/env/bin/python3
LOG=$WT/scripts/rfd3_port/p27_sweep.log
MODE="${1:?usage: run_p27_sweep.sh <ab|full> <fixture>...}"; shift
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:tt-bio-rfdiffusion3-largedesign-gap-p27

args_for() {
  case "$1" in
    iai40)  echo --contig "A1-10,20,A31-40" ;;
    iai80)  echo --contig "A1-10,60,A31-40" ;;
    iai150) echo --contig "A1-10,130,A31-40" ;;
    iai250) echo --contig "A1-10,230,A31-40" ;;
    mpro)   echo --spec scripts/rfd3_port/parity_artifacts/enzyme_mpro/spec.json ;;
    *) echo "unknown fixture $1" >&2; return 1 ;;
  esac
}

run_one() {  # <variant> <tree> <fixture> <rep> <extra args...>
  local variant="$1" tree="$2" fixture="$3" rep="$4"; shift 4
  echo "--- $fixture $variant rep$rep start $(date -Is) ---" >> "$LOG"
  ( cd "$tree" && PYTHONPATH="$tree" "$PY" \
      "$tree/scripts/rfd3_port/p27_real_design_timing.py" \
      --tag "$fixture/$variant/rep$rep" "$@" $(cd "$WT" && args_for "$fixture") \
  ) 2>&1 | grep -E "^fixture:|^RESULT " | sed "s/^/[$fixture $variant rep$rep] /" >> "$LOG"
  echo "--- $fixture $variant rep$rep done $(date -Is) ---" >> "$LOG"
}

for fixture in "$@"; do
  case "$MODE" in
    ab)
      # 20 timed steps + the priced build, both trees, order reversed on rep 2
      run_one base "$REF" "$fixture" 1 --timesteps 21 --batches 1 8 --steady-pass
      run_one fix  "$WT"  "$fixture" 1 --timesteps 21 --batches 1 8 --steady-pass
      run_one fix  "$WT"  "$fixture" 2 --timesteps 21 --batches 1 8 --steady-pass
      run_one base "$REF" "$fixture" 2 --timesteps 21 --batches 1 8 --steady-pass
      ;;
    full)
      # a real 200-timestep design, one-time build inside the timed region
      run_one fix "$WT" "$fixture" 1 --timesteps 200 --batches 1 8
      ;;
    anchor)
      # both trees at a real 200 timesteps, to check the composed number against an
      # end-to-end one rather than trusting build + 199 * steady on its own
      run_one base "$REF" "$fixture" 200 --timesteps 200 --batches 1
      run_one fix  "$WT"  "$fixture" 200 --timesteps 200 --batches 1
      ;;
    *) echo "unknown mode $MODE" >&2; exit 1 ;;
  esac
done
echo "=== $MODE sweep complete $(date -Is) ===" >> "$LOG"
