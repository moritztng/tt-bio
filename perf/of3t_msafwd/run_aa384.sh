#!/usr/bin/env bash
# of3t-msafwd: re-take of3t-msaamp's whole-module device arm (msa_instrument.py, training tape) at
# 384 tokens and at the 64 crop, through run_ours.sh (host_quiet, AICLK DURING, board stamp).
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-msafwd; cd "$W" || exit 1
S=/home/ttuser/of3t_msaamp
for spec in "aa384 $S/cap043b/boundary_msa_module.pt" "aa64 $S/crop64.pt"; do
  set -- $spec
  bash perf/of3t_msafwd/run_ours.sh "$1" perf/of3t_auxheads/msa_instrument.py --boundary "$2" \
    --reference-grads /home/ttuser/of3t-campaign-refs/bundle_min_043/grads_f64_043.pt \
    --out "perf/of3t_msafwd/INSTR_$1.json" 2>&1 | grep -E "^===|FWD|Error|error" | head -8
done
