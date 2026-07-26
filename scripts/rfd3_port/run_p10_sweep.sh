#!/bin/bash
# p10 five-fixture best-vs-best sweep, A/B'd against the pre-change tree.
#
# Both variants run back to back on the same card for each fixture, so thermal
# drift cannot masquerade as a speedup. The baseline is a `git archive` of the
# pre-change commit (see verify_mask_template_parity.py for how it is built)
# rather than a `git stash`, so an interrupted run never leaves a dirty tree.
#
#   scripts/rfd3_port/run_p10_sweep.sh <fixture> [<fixture> ...]
# fixtures: iai40 iai80 iai150 mpro iai250
set -u
WT=/home/moritz/.coworker/wt/tt-bio-rfdiffusion3-batch-perf-p10
REF=/tmp/p10_ref
PY=/home/moritz/tt-bio/env/bin/python3
LOG=$WT/scripts/rfd3_port/p10_sweep.log
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:tt-bio-rfdiffusion3-batch-perf-p10
export RFD3_TRACE_DECODER=1
export TT_BIO_TRACE_REGION_SIZE=268435456

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

run_one() {  # <variant> <tree> <fixture> <batches...>
  local variant="$1" tree="$2" fixture="$3"; shift 3
  echo "--- $fixture $variant start $(date -Is) ---" >> "$LOG"
  ( cd "$tree" && PYTHONPATH="$tree" "$PY" \
      "$tree/scripts/rfd3_port/bench_batch_designs_per_sec.py" \
      --timesteps 40 --batches "$@" $(cd "$WT" && args_for "$fixture") \
    ) 2>&1 | grep -E "^fixture:|^[0-9]+ " | sed "s/^/[$fixture $variant] /" >> "$LOG"
  echo "--- $fixture $variant done rc=$? $(date -Is) ---" >> "$LOG"
}

for fixture in "$@"; do
  # the mpro spec path is relative to the worktree; give the ref tree its copy
  if [ "$fixture" = mpro ] && [ ! -f "$REF/scripts/rfd3_port/parity_artifacts/enzyme_mpro/spec.json" ]; then
    mkdir -p "$REF/scripts/rfd3_port/parity_artifacts/enzyme_mpro"
    cp -r "$WT/scripts/rfd3_port/parity_artifacts/enzyme_mpro/." \
          "$REF/scripts/rfd3_port/parity_artifacts/enzyme_mpro/"
  fi
  run_one base "$REF" "$fixture" 1 8
  run_one fix  "$WT"  "$fixture" 1 8
done
tail -n 40 "$LOG"
