#!/usr/bin/env bash
# One lock, every arm. Ordered safest-first: the screen modes and the CB-depth arms cannot deadlock,
# mode 14 changes the reader/compute handshake and is therefore last.
set -u
cd /home/ttuser/.coworker/wt/relion-projection-optimize
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:relion-projection-optimize
LOG=perf/relion-projection-optimize/arms.log
mkdir -p perf/relion-projection-optimize
run() {
  echo "=== ARM: $* ===" | tee -a "$LOG"
  timeout 900 python3 projprobe/fslice_e2e.py "$@" 2>&1 | tee -a "$LOG"
  echo "--- rc=${PIPESTATUS[0]} ---" | tee -a "$LOG"
}
run 256 --reps 5 --split                          # arm 0, baseline + per-stage
run 256 --reps 5 --skip-parity                    # arm 0, A/A repeat
run 256 --reps 5 --skip-parity --fid HiFi2        # lever F
run 256 --reps 5 --skip-parity --cbtil 8          # lever O(a)
run 256 --reps 5 --skip-parity --cbtil 16 --cbsrc 8
run 256 --reps 5 --skip-parity --mode 5           # screen: one assembly per block, no 2nd pass
run 256 --reps 5 --skip-parity --mode 6           # screen: mode 5 minus the per-block tilize
run 256 --reps 5 --skip-parity --mode 0           # screen: reader + tilize only
run 256 --reps 5 --mode 14 --dscale 1.1228        # LEVER B
run 256 --reps 5 --skip-parity --mode 14 --dscale 1.4142
