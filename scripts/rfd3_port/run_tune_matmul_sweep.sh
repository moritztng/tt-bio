#!/bin/bash
# Interleaved A/B of RFD3_TUNE_MATMUL across the five design fixtures.
#
# base and fix run back to back on the same card for each fixture and round, so thermal drift
# cannot masquerade as a speedup (a fix-then-base ordering overstated p14's win ~2x on card
# warmth). Both variants are THIS tree -- the calibrated-matmul code is inert with the flag
# unset, and dump_forward_for_crosstree_parity.py has already shown this tree bit-identical to
# 01cc93c6f with the flag on -- so the A/B isolates the flag and nothing else.
#
# The order ALTERNATES by round, because back-to-back alone is not enough: whichever variant runs
# second inherits a warmer card and reads ~5% slower. Measured with D=1, where the flag changes
# nothing at all and the two variants execute identical code -- fixed base-then-fix ordering
# still showed "fix" slower in 5 of 5 rounds. Use an even number of rounds so the bias cancels.
#
#   scripts/rfd3_port/run_tune_matmul_sweep.sh [--rounds N] <fixture> [<fixture> ...]
# fixtures: iai40 iai80 iai150 mpro iai250
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/moritz/tt-bio/env/bin/python3
LOG=$WT/scripts/rfd3_port/tune_matmul_sweep.log
TIMESTEPS=${TIMESTEPS:-20}
ROUNDS=1
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:tt-bio-rfdiffusion3-batch-perf-p16
export RFD3_TRACE_DECODER=1
export TT_BIO_TRACE_REGION_SIZE=268435456

if [ "${1:-}" = --rounds ]; then ROUNDS=$2; shift 2; fi

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

run_one() {  # <variant> <round> <fixture>
  local variant="$1" round="$2" fixture="$3" tag raw
  tag="[$fixture r$round $variant]"
  local tune=0; [ "$variant" = fix ] && tune=1
  raw=$(mktemp)
  RFD3_TUNE_MATMUL=$tune PYTHONPATH="$WT" "$PY" \
      "$WT/scripts/rfd3_port/bench_batch_designs_per_sec.py" \
      --timesteps "$TIMESTEPS" --batches 1 8 $(args_for "$fixture") > "$raw" 2>&1
  local rc=$?
  # Filtering the run's output through grep and throwing the rest away makes a crashed run look
  # exactly like a run that produced no rows. One iai40 round vanished that way. Keep the tail.
  if [ $rc -ne 0 ] || ! grep -qE "^[0-9]+ " "$raw"; then
    { echo "$tag FAILED rc=$rc -- last 20 lines:"; tail -20 "$raw"; } | tee -a "$LOG"
  else
    grep -E "^[0-9]+ " "$raw" | sed "s/^/$tag /" | tee -a "$LOG"
  fi
  rm -f "$raw"
}

echo "=== sweep start $(date -Is) timesteps=$TIMESTEPS rounds=$ROUNDS fixtures=$* ===" >> "$LOG"
for round in $(seq 1 "$ROUNDS"); do
  for fixture in "$@"; do
    if [ $((round % 2)) -eq 1 ]; then
      run_one base "$round" "$fixture"; run_one fix  "$round" "$fixture"
    else
      run_one fix  "$round" "$fixture"; run_one base "$round" "$fixture"
    fi
  done
done
echo "=== sweep done $(date -Is) ===" >> "$LOG"
