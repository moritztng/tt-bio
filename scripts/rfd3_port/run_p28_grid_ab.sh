#!/usr/bin/env bash
# Interleaved A/B of RFD3_FAST_GRID=0/1 in ONE tree.
#
# One tree means the two arms differ only by an env var, so this A/B cannot be a harness
# diff -- strictly stronger than the `git archive` rig p10/p14/p27 used. Arm order is
# flipped between rounds because alternating alone does not cancel thermal bias on this
# card (rfd3-p14). The kernel cache is warmed by the harness's own 4-step warmup, and the
# first run of the whole sweep is discarded (rfd3-baseline-seed-cold-cache-trap).
set -uo pipefail

TREE="${TREE:-/tmp/p28wt}"
PY="${PY:-/home/moritz/tt-bio/env/bin/python3}"
CARD="${CARD:-0}"
HOLDER="worker:tt-bio-rfdiffusion3-largedesign-gap-p28"
STEPS="${STEPS:-20}"
BATCHES="${BATCHES:-1}"
CONTIG="${CONTIG:-A1-10,230,A31-40}"
ROUNDS="${ROUNDS:-2}"
LOG="${LOG:-/tmp/p28/grid_ab.log}"
mkdir -p "$(dirname "$LOG")"

run_arm() {  # $1 = 0|1, $2 = round
  RFD3_FAST_GRID="$1" TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_HOLDER="$HOLDER" \
  PYTHONPATH="$TREE" "$PY" "$TREE/scripts/rfd3_port/p27_real_design_timing.py" \
    --timesteps "$STEPS" --batches $BATCHES --contig "$CONTIG" \
    --tag "fastgrid=$1 round=$2" 2>&1 | grep -E '^(RESULT|fixture)'
}

echo "== discard run (cold kernel cache) ==" | tee -a "$LOG"
run_arm 0 warm >>"$LOG" 2>&1

for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="0 1"; else order="1 0"; fi
  for arm in $order; do
    echo "== round $r arm fastgrid=$arm ==" | tee -a "$LOG"
    run_arm "$arm" "$r" | tee -a "$LOG"
  done
done
echo "log: $LOG"
