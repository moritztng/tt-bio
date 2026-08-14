#!/usr/bin/env bash
# The arm batch, re-homed on pc card 0 (the qb1 lock never freed; arms.sh is kept for provenance).
# One benchlock acquisition for every arm. Roofs go FIRST: the floor has to be this card's, not qb1's.
# Ordered safest-first, and mode 14 sits second-to-last because it is the only arm that changes the
# reader/compute handshake -- a generic_op deadlock costs a card reset and everything after it.
set -u
cd /home/moritz/.coworker/wt/relion-projection-optimize
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:relion-projection-optimize
PY=/home/moritz/tt-bio/env/bin/python3
LOG=perf/relion-projection-optimize/arms_pc.log
mkdir -p perf/relion-projection-optimize
: > "$LOG"
run() {
  echo "=== ARM: $* ===" | tee -a "$LOG"
  timeout 1200 $PY "$@" 2>&1 | tee -a "$LOG"
  echo "--- rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S) ---" | tee -a "$LOG"
}
run projprobe/b0_roofs.py                                          # this card's roofs -> the floor
run projprobe/fslice_e2e.py 256 --reps 5 --split                   # arm 0, baseline + per-stage walls
run projprobe/fslice_e2e.py 256 --reps 5 --skip-parity             # arm 0 again, the A/A noise floor
run projprobe/fslice_e2e.py 256 --reps 5 --skip-parity --mode 0    # screen: reader + tilize only
run projprobe/fslice_e2e.py 256 --reps 5 --skip-parity --fid HiFi2 # lever F
run projprobe/fslice_e2e.py 256 --reps 5 --skip-parity --cbtil 8   # lever O(a)
run projprobe/fslice_e2e.py 256 --reps 5 --mode 14 --dscale 1.1228 # LEVER B, parity ON
run projprobe/fslice_e2e.py 256 --reps 5 --skip-parity --mode 14 --dscale 1.4142
run projprobe/fslice_e2e.py 256 --reps 5 --skip-parity --mode 5
run projprobe/fslice_e2e.py 256 --reps 5 --skip-parity --mode 6
echo "=== BATCH DONE $(date -u +%FT%TZ) ===" | tee -a "$LOG"
