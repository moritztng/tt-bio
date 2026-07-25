#!/bin/bash
set -u
WT=/home/ttuser/.coworker/wt/tt-bio-rfdiffusion3-batch-perf-p4
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:tt-bio-rfdiffusion3-batch-perf-p4
export RFD3_TRACE_DECODER=1
export TT_BIO_TRACE_REGION_SIZE=268435456
export PYTHONPATH=$WT
LOG=$WT/scripts/rfd3_port/p4_sweep.log
echo "=== sweep start $(date -Is) ===" >> "$LOG"
run() {
  local label="$1"; shift
  echo "--- $label $(date -Is) ---" >> "$LOG"
  python3 scripts/rfd3_port/bench_batch_designs_per_sec.py "$@" >> "$LOG" 2>&1
  echo "--- $label done rc=$? ---" >> "$LOG"
}
run "mpro" --timesteps 200 --batches 1 8 --spec scripts/rfd3_port/parity_artifacts/enzyme_mpro/spec.json
echo "=== sweep end $(date -Is) ===" >> "$LOG"
