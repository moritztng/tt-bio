#!/usr/bin/env bash
# Batch 2: the parity gate now sees the arm under test, and the box sweep the deliverable owes.
set -u
cd /home/moritz/.coworker/wt/relion-projection-optimize
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:relion-projection-optimize
PY=/home/moritz/tt-bio/env/bin/python3
LOG=perf/relion-projection-optimize/arms_pc2.log
: > "$LOG"
run() {
  echo "=== ARM: $* ===" | tee -a "$LOG"
  timeout 1200 $PY "$@" 2>&1 | tee -a "$LOG"
  echo "--- rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S) ---" | tee -a "$LOG"
}
run projprobe/fslice_e2e.py 256 --reps 5 --fid HiFi4                 # parity control, gate now live
run projprobe/fslice_e2e.py 256 --reps 5 --fid HiFi2                 # lever F, parity ON and MODE-AWARE
run projprobe/fslice_e2e.py 256 --reps 5 --fid LoFi                  # lever F, the bracket
run projprobe/fslice_e2e.py 384 --reps 5 --skip-parity               # arm 0, box 384
run projprobe/fslice_e2e.py 384 --reps 5 --skip-parity --mode 14 --dscale 1.1228
run projprobe/fslice_e2e.py 512 --reps 5 --skip-parity               # arm 0, box 512
run projprobe/fslice_e2e.py 512 --reps 5 --skip-parity --mode 14 --dscale 1.1228
run projprobe/fslice_e2e.py 256 --reps 5 --skip-parity --mode 14 --dscale 1.1228 --fid HiFi2  # B+F
echo "=== BATCH2 DONE $(date -u +%FT%TZ) ===" | tee -a "$LOG"
